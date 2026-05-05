/**
 * Stripe webhook handler.
 *
 * IMPORTANT: Stripe signature verification requires the EXACT raw request body,
 * byte-for-byte. firebase-functions onRequest exposes the raw payload on
 * `req.rawBody` (a Buffer). Using `req.body` after JSON parsing produces a
 * different byte sequence and signature verification fails.
 *
 * Flow on checkout.session.completed:
 *   1. Verify signature using STRIPE_WEBHOOK_SECRET.
 *   2. Generate a unique CLLDR-XXXX-XXXX-XXXX-XXXX license key.
 *   3. Persist to Firestore at licenses/{key}. The buyer retrieves the key
 *      via the dashboard (queries licenses where email == auth.token.email).
 *   4. Return 200.
 *
 * Idempotency: Stripe retries delivery on non-2xx responses. We dedupe on
 * stripeSessionId to avoid issuing two keys for the same purchase.
 */

import { onRequest, Request } from "firebase-functions/v2/https";
import { logger } from "firebase-functions/v2";
import * as admin from "firebase-admin";
import Stripe from "stripe";
import { Response } from "express";

import { generateLicenseKey } from "./licenseKey";

const MAX_KEY_GENERATION_ATTEMPTS = 5;

let cachedStripe: Stripe | null = null;

function getStripe(): Stripe {
  if (cachedStripe) return cachedStripe;
  // Stripe SDK requires an API key only when making outbound calls.
  // For webhook verification only, an empty string works, but we wire the
  // real key (from env) so the SDK is correctly configured if we ever
  // call other Stripe APIs from here.
  const key = process.env.STRIPE_SECRET_KEY ?? "sk_placeholder_for_webhook_only";
  cachedStripe = new Stripe(key, {
    apiVersion: "2025-02-24.acacia",
  });
  return cachedStripe;
}

function getWebhookSecret(): string {
  const secret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!secret) {
    throw new Error("STRIPE_WEBHOOK_SECRET is not set in the environment.");
  }
  return secret;
}

async function findExistingLicenseForSession(
  sessionId: string
): Promise<FirebaseFirestore.QueryDocumentSnapshot | null> {
  const db = admin.firestore();
  const snapshot = await db
    .collection("licenses")
    .where("stripeSessionId", "==", sessionId)
    .limit(1)
    .get();
  if (snapshot.empty) return null;
  return snapshot.docs[0];
}

async function issueLicense(
  email: string,
  sessionId: string
): Promise<string> {
  const db = admin.firestore();

  for (let attempt = 0; attempt < MAX_KEY_GENERATION_ATTEMPTS; attempt += 1) {
    const key = generateLicenseKey();
    const docRef = db.collection("licenses").doc(key);
    try {
      await db.runTransaction(async (tx) => {
        const existing = await tx.get(docRef);
        if (existing.exists) {
          throw new Error("collision");
        }
        tx.set(docRef, {
          key,
          email,
          stripeSessionId: sessionId,
          status: "active",
          createdAt: admin.firestore.FieldValue.serverTimestamp(),
        });
      });
      return key;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (message !== "collision") throw err;
      logger.warn("License key collision, retrying", { attempt, key });
    }
  }
  throw new Error(
    `Could not generate a unique license key after ${MAX_KEY_GENERATION_ATTEMPTS} attempts.`
  );
}

async function handleCheckoutSessionCompleted(
  session: Stripe.Checkout.Session
): Promise<void> {
  const sessionId = session.id;
  const email =
    session.customer_details?.email ??
    session.customer_email ??
    null;

  if (!email) {
    logger.error("Stripe session missing customer email", { sessionId });
    throw new Error("No customer email on Stripe checkout session.");
  }

  // Idempotency: if we already issued a key for this session, skip.
  const existing = await findExistingLicenseForSession(sessionId);
  if (existing) {
    logger.info("Duplicate webhook for session, skipping issuance", {
      sessionId,
      key: existing.id,
    });
    return;
  }

  const key = await issueLicense(email, sessionId);
  logger.info("License issued", { key, email, sessionId });
}

export const stripeWebhook = onRequest(
  {
    region: "us-central1",
    invoker: "public",
    cors: false,
  },
  async (req: Request, res: Response): Promise<void> => {
    if (req.method !== "POST") {
      res.status(405).send("Method Not Allowed");
      return;
    }

    const signature = req.headers["stripe-signature"];
    if (!signature || typeof signature !== "string") {
      logger.warn("Webhook missing Stripe-Signature header");
      res.status(400).send("Missing Stripe-Signature header");
      return;
    }

    let webhookSecret: string;
    try {
      webhookSecret = getWebhookSecret();
    } catch (err) {
      logger.error("Webhook secret not configured", err);
      res.status(500).send("Webhook secret not configured");
      return;
    }

    let event: Stripe.Event;
    try {
      // req.rawBody is provided by the Firebase Functions runtime and contains
      // the unparsed request body as a Buffer. This is required for signature
      // verification. DO NOT use req.body here.
      const rawBody: Buffer = (req as Request & { rawBody: Buffer }).rawBody;
      event = getStripe().webhooks.constructEvent(
        rawBody,
        signature,
        webhookSecret
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.warn("Stripe signature verification failed", { message });
      res.status(400).send(`Webhook signature verification failed: ${message}`);
      return;
    }

    logger.info("Stripe event received", { type: event.type, id: event.id });

    try {
      switch (event.type) {
        case "checkout.session.completed": {
          const session = event.data.object as Stripe.Checkout.Session;
          await handleCheckoutSessionCompleted(session);
          break;
        }
        default:
          logger.info("Ignoring unhandled event type", { type: event.type });
      }
      res.status(200).json({ received: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.error("Error processing Stripe event", {
        type: event.type,
        id: event.id,
        message,
      });
      res.status(500).send(`Error processing event: ${message}`);
    }
  }
);
