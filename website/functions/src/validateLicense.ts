/**
 * License validation endpoint.
 *
 * POST { key: string } -> { valid: boolean, email?: string, status?: string }
 *
 * CORS: open to all origins. This endpoint is called from the desktop binary
 * (no Origin header) and from the /pro page on the website. Returns no
 * sensitive details on invalid keys to avoid enumeration leaks.
 */

import { onRequest, Request } from "firebase-functions/v2/https";
import { logger } from "firebase-functions/v2";
import * as admin from "firebase-admin";
import { Response } from "express";

import { LICENSE_KEY_REGEX } from "./licenseKey";

interface ValidateRequestBody {
  key?: unknown;
}

interface ValidateResponseBody {
  valid: boolean;
  email?: string;
  status?: string;
}

function applyCors(res: Response): void {
  res.set("Access-Control-Allow-Origin", "*");
  res.set("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.set("Access-Control-Allow-Headers", "Content-Type");
  res.set("Access-Control-Max-Age", "3600");
}

function parseKey(body: unknown): string | null {
  if (!body || typeof body !== "object") return null;
  const { key } = body as ValidateRequestBody;
  if (typeof key !== "string") return null;
  const trimmed = key.trim().toUpperCase();
  if (!LICENSE_KEY_REGEX.test(trimmed)) return null;
  return trimmed;
}

export const validateLicense = onRequest(
  {
    region: "us-central1",
    invoker: "public",
    cors: false, // we set headers manually for full control
  },
  async (req: Request, res: Response): Promise<void> => {
    applyCors(res);

    if (req.method === "OPTIONS") {
      res.status(204).send("");
      return;
    }

    if (req.method !== "POST") {
      res.status(405).json({ valid: false } as ValidateResponseBody);
      return;
    }

    // req.body is the parsed JSON payload (firebase-functions v2 parses by
    // default for content-type application/json).
    const key = parseKey(req.body);
    if (!key) {
      res.status(200).json({ valid: false } as ValidateResponseBody);
      return;
    }

    try {
      const db = admin.firestore();
      const docRef = db.collection("licenses").doc(key);
      const snap = await docRef.get();

      if (!snap.exists) {
        res.status(200).json({ valid: false } as ValidateResponseBody);
        return;
      }

      const data = snap.data() ?? {};
      const status = typeof data.status === "string" ? data.status : "unknown";
      const email = typeof data.email === "string" ? data.email : undefined;
      const isActive = status === "active";

      // Touch the validatedAt timestamp on every valid lookup. Fire-and-forget
      // so a slow write does not block the response.
      if (isActive) {
        docRef
          .update({ validatedAt: admin.firestore.FieldValue.serverTimestamp() })
          .catch((err) => {
            logger.warn("Failed to update validatedAt", { key, err });
          });
      }

      const response: ValidateResponseBody = {
        valid: isActive,
        status,
      };
      if (email) response.email = email;

      res.status(200).json(response);
    } catch (err) {
      logger.error("validateLicense error", { err });
      res.status(500).json({ valid: false } as ValidateResponseBody);
    }
  }
);
