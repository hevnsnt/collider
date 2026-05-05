/**
 * License key generator.
 *
 * Format: CLLDR-XXXX-XXXX-XXXX-XXXX where each X is uppercase A-Z or 0-9.
 *
 * Uses crypto.randomInt for cryptographically strong randomness so keys
 * cannot be predicted from one another.
 */

import { randomInt } from "crypto";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const GROUP_LEN = 4;
const GROUP_COUNT = 4;

export const LICENSE_KEY_REGEX =
  /^CLLDR-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$/;

function randomGroup(): string {
  let out = "";
  for (let i = 0; i < GROUP_LEN; i += 1) {
    out += ALPHABET[randomInt(0, ALPHABET.length)];
  }
  return out;
}

export function generateLicenseKey(): string {
  const groups: string[] = [];
  for (let i = 0; i < GROUP_COUNT; i += 1) {
    groups.push(randomGroup());
  }
  return `CLLDR-${groups.join("-")}`;
}
