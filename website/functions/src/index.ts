/**
 * theCollider Pro - Cloud Functions entrypoint
 *
 * Exposes:
 *   - stripeWebhook: handles Stripe checkout.session.completed events,
 *     issues a license key, persists it to Firestore, and emails the buyer.
 *   - validateLicense: looks up a license key and returns its status.
 *
 * Region: us-central1 (Firebase default).
 */

import { setGlobalOptions } from "firebase-functions/v2";
import * as admin from "firebase-admin";

// Initialize Admin SDK exactly once.
if (admin.apps.length === 0) {
  admin.initializeApp();
}

// Conservative defaults. Webhook should be snappy; validation is a single read.
setGlobalOptions({
  region: "us-central1",
  maxInstances: 10,
  timeoutSeconds: 30,
  memory: "256MiB",
});

export { stripeWebhook } from "./stripeWebhook";
export { validateLicense } from "./validateLicense";
