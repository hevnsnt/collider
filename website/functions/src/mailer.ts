/**
 * Transactional email delivery via Gmail SMTP using nodemailer.
 *
 * Credentials come from env: GMAIL_USER and GMAIL_APP_PASSWORD.
 * GMAIL_APP_PASSWORD must be a Google App Password, not the account password.
 */

import * as nodemailer from "nodemailer";
import { logger } from "firebase-functions/v2";

let cachedTransporter: nodemailer.Transporter | null = null;

function getTransporter(): nodemailer.Transporter {
  if (cachedTransporter) {
    return cachedTransporter;
  }
  const user = process.env.GMAIL_USER;
  const pass = process.env.GMAIL_APP_PASSWORD;
  if (!user || !pass) {
    throw new Error(
      "Gmail credentials missing. Set GMAIL_USER and GMAIL_APP_PASSWORD."
    );
  }
  cachedTransporter = nodemailer.createTransport({
    service: "gmail",
    auth: { user, pass },
  });
  return cachedTransporter;
}

export interface LicenseEmailParams {
  to: string;
  licenseKey: string;
}

export async function sendLicenseEmail(params: LicenseEmailParams): Promise<void> {
  const { to, licenseKey } = params;
  const transporter = getTransporter();
  const from = process.env.GMAIL_USER;

  const text = [
    "Thank you for purchasing theCollider Pro.",
    "",
    "Your license key:",
    licenseKey,
    "",
    "Activate it by running:",
    `  ./collider --activate ${licenseKey}`,
    "",
    "Keep this key safe. It is not tied to a specific machine,",
    "but you are responsible for keeping it private.",
    "",
    "Documentation: https://collisionprotocol.com/docs",
    "Support: hevnsnt@gmail.com",
    "",
    "Happy hunting,",
    "Collision Protocol",
  ].join("\n");

  const html = `
    <div style="font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace; background: #0a0a0a; color: #e0e0e0; padding: 32px; max-width: 640px; margin: 0 auto;">
      <h1 style="color: #00ffff; font-size: 24px; margin: 0 0 16px;">theCollider Pro</h1>
      <p style="color: #e0e0e0; line-height: 1.6;">Thank you for your purchase. Your license key is below.</p>
      <div style="background: #111111; border: 1px solid #333333; border-radius: 8px; padding: 24px; margin: 24px 0; text-align: center;">
        <div style="color: #888888; font-size: 12px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px;">License Key</div>
        <div style="color: #00ffff; font-size: 20px; letter-spacing: 2px; font-weight: bold;">${licenseKey}</div>
      </div>
      <p style="color: #e0e0e0; line-height: 1.6;">Activate it by running:</p>
      <pre style="background: #111111; border: 1px solid #333333; border-radius: 6px; padding: 16px; color: #ffb000; overflow-x: auto;">./collider --activate ${licenseKey}</pre>
      <p style="color: #888888; font-size: 13px; line-height: 1.6;">Keep this key safe. It is not tied to a specific machine, but you are responsible for keeping it private.</p>
      <hr style="border: none; border-top: 1px solid #333333; margin: 32px 0;" />
      <p style="color: #555555; font-size: 12px;">
        Documentation: <a href="https://collisionprotocol.com/docs" style="color: #00ffff;">collisionprotocol.com/docs</a><br />
        Support: <a href="mailto:hevnsnt@gmail.com" style="color: #00ffff;">hevnsnt@gmail.com</a>
      </p>
    </div>
  `.trim();

  try {
    await transporter.sendMail({
      from: `"Collision Protocol" <${from}>`,
      to,
      subject: "Your theCollider Pro License Key",
      text,
      html,
    });
    logger.info("License email sent", { to });
  } catch (err) {
    logger.error("Failed to send license email", { to, err });
    throw err;
  }
}
