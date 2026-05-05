"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Check,
  X,
  Zap,
  Shield,
  ArrowRight,
  KeyRound,
  Loader2,
  CheckCircle2,
  XCircle,
  Cpu,
  BookOpen,
  Brain,
  ListOrdered,
} from "lucide-react";

import { Navbar, Footer } from "@/components/layout";
import { Button, Input, Badge } from "@/components/ui";
import { cn } from "@/lib/utils";

const STRIPE_CHECKOUT_URL =
  "https://buy.stripe.com/6oU9AK49ybbx6Ci8Xldby00";

const VALIDATE_LICENSE_URL =
  "https://us-central1-clawcfg-741ab.cloudfunctions.net/validateLicense";

const LICENSE_KEY_PATTERN = /^CLLDR-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$/;

interface FeatureRow {
  name: string;
  free: boolean | string;
  pro: boolean | string;
  highlight?: boolean;
  icon?: React.ComponentType<{ className?: string }>;
}

const featureRows: FeatureRow[] = [
  {
    name: "Puzzle solving (Kangaroo, BSGS)",
    free: true,
    pro: true,
    icon: Cpu,
  },
  {
    name: "JLP pool client",
    free: true,
    pro: true,
    icon: Zap,
  },
  {
    name: "Open source core",
    free: true,
    pro: true,
    icon: BookOpen,
  },
  {
    name: "Brainwallet scanning pipeline",
    free: false,
    pro: true,
    highlight: true,
    icon: Brain,
  },
  {
    name: "Custom wordlists & dictionaries",
    free: false,
    pro: true,
    highlight: true,
    icon: ListOrdered,
  },
  {
    name: "GPU rule engine (PCFG, mangling)",
    free: false,
    pro: true,
    highlight: true,
    icon: Cpu,
  },
  {
    name: "Solo solving mode (no pool fee)",
    free: false,
    pro: true,
    highlight: true,
    icon: Shield,
  },
];

interface ValidateResponse {
  valid: boolean;
  email?: string;
  status?: string;
}

type ValidationState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "valid"; email?: string; status?: string }
  | { kind: "invalid" }
  | { kind: "error"; message: string };

function CellValue({ value }: { value: boolean | string }) {
  if (value === true) {
    return (
      <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-success/10 border border-success/30">
        <Check className="w-4 h-4 text-success" />
      </span>
    );
  }
  if (value === false) {
    return (
      <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-error/5 border border-error/20">
        <X className="w-4 h-4 text-error/60" />
      </span>
    );
  }
  return <span className="font-mono text-sm text-foreground">{value}</span>;
}

export default function ProPage() {
  const [licenseKey, setLicenseKey] = useState("");
  const [validation, setValidation] = useState<ValidationState>({
    kind: "idle",
  });

  async function handleValidate(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const trimmed = licenseKey.trim().toUpperCase();
    if (!LICENSE_KEY_PATTERN.test(trimmed)) {
      setValidation({
        kind: "error",
        message: "Format: CLLDR-XXXX-XXXX-XXXX-XXXX (A-Z, 0-9)",
      });
      return;
    }

    setValidation({ kind: "loading" });
    try {
      const res = await fetch(VALIDATE_LICENSE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: trimmed }),
      });

      if (!res.ok) {
        setValidation({
          kind: "error",
          message: `Server returned ${res.status}. Try again in a moment.`,
        });
        return;
      }

      const data: ValidateResponse = await res.json();
      if (data.valid) {
        setValidation({
          kind: "valid",
          email: data.email,
          status: data.status,
        });
      } else {
        setValidation({ kind: "invalid" });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Network error";
      setValidation({ kind: "error", message });
    }
  }

  return (
    <>
      <Navbar />
      <main className="bg-background">
        {/* HERO */}
        <section className="relative pt-32 pb-20 lg:pt-40 lg:pb-28 overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-cyan/5 via-background to-amber/5" />
          <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-20" />
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan/10 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-amber/10 rounded-full blur-3xl" />

          <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="inline-flex"
            >
              <Badge variant="amber" className="mb-6">
                <Shield className="w-3 h-3 mr-1.5" />
                One-time purchase &middot; No subscription
              </Badge>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
              className="text-4xl sm:text-5xl lg:text-7xl font-mono font-bold text-foreground mb-6"
            >
              theCollider <span className="text-cyan">Pro</span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="text-lg sm:text-xl text-foreground-secondary max-w-2xl mx-auto mb-10"
            >
              Unlock the full brainwallet pipeline. Solo solving.
              Maximum throughput.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <Link
                href={STRIPE_CHECKOUT_URL}
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="amber" size="lg">
                  Buy Now &mdash; $49.99
                  <ArrowRight className="w-5 h-5" />
                </Button>
              </Link>
              <Link href="#validator">
                <Button variant="secondary" size="lg">
                  <KeyRound className="w-5 h-5" />
                  Validate License
                </Button>
              </Link>
            </motion.div>
          </div>
        </section>

        {/* COMPARISON */}
        <section className="py-20 lg:py-28 bg-background-secondary relative">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-12">
              <motion.span
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="inline-block font-mono text-xs uppercase tracking-widest text-cyan mb-4"
              >
                Free vs Pro
              </motion.span>
              <motion.h2
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 }}
                className="text-3xl sm:text-4xl font-mono font-bold text-foreground mb-4"
              >
                What you <span className="text-cyan">unlock</span>
              </motion.h2>
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
                className="text-foreground-secondary max-w-2xl mx-auto"
              >
                Free covers the core solver. Pro unlocks the brainwallet
                pipeline, custom wordlists, the GPU rule engine, and solo
                mode.
              </motion.p>
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-background-tertiary to-background-secondary"
            >
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-5 px-6 font-mono text-xs uppercase tracking-wider text-foreground-secondary">
                      Capability
                    </th>
                    <th className="text-center py-5 px-6 font-mono text-xs uppercase tracking-wider text-foreground-secondary">
                      Free
                    </th>
                    <th className="text-center py-5 px-6 font-mono text-xs uppercase tracking-wider text-amber bg-amber/5">
                      Pro
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {featureRows.map((row) => {
                    const Icon = row.icon;
                    return (
                      <tr
                        key={row.name}
                        className={cn(
                          "border-b border-border/40 last:border-b-0",
                          row.highlight && "bg-amber/5"
                        )}
                      >
                        <td className="py-4 px-6 text-sm text-foreground">
                          <div className="flex items-center gap-3">
                            {Icon && (
                              <Icon
                                className={cn(
                                  "w-4 h-4 flex-shrink-0",
                                  row.highlight
                                    ? "text-amber"
                                    : "text-foreground-muted"
                                )}
                              />
                            )}
                            <span>{row.name}</span>
                          </div>
                        </td>
                        <td className="text-center py-4 px-6">
                          <div className="flex items-center justify-center">
                            <CellValue value={row.free} />
                          </div>
                        </td>
                        <td
                          className={cn(
                            "text-center py-4 px-6",
                            "bg-amber/5"
                          )}
                        >
                          <div className="flex items-center justify-center">
                            <CellValue value={row.pro} />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </motion.div>
          </div>
        </section>

        {/* PRICE CARD */}
        <section className="py-20 lg:py-28 relative overflow-hidden">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-amber/5 rounded-full blur-3xl" />

          <div className="relative z-10 max-w-2xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="bg-gradient-to-br from-background-tertiary to-background-secondary border border-amber/30 rounded-2xl p-8 sm:p-12 text-center shadow-glow-amber"
            >
              <Badge variant="amber" className="mb-6">
                LIFETIME ACCESS
              </Badge>

              <div className="mb-8">
                <div className="flex items-baseline justify-center gap-2">
                  <span className="text-2xl text-foreground-muted font-mono">
                    $
                  </span>
                  <span className="text-7xl sm:text-8xl font-mono font-bold text-amber">
                    49.99
                  </span>
                </div>
                <p className="font-mono text-sm text-foreground-secondary uppercase tracking-widest mt-2">
                  USD
                </p>
              </div>

              <h2 className="text-2xl sm:text-3xl font-mono font-bold text-foreground mb-4">
                One-time purchase
              </h2>
              <p className="text-foreground-secondary mb-8">
                No subscription. No machine lock. Use your key on every rig you
                own.
              </p>

              <Link
                href={STRIPE_CHECKOUT_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-block w-full sm:w-auto"
              >
                <Button variant="amber" size="lg" className="w-full sm:w-auto">
                  Buy Now
                  <ArrowRight className="w-5 h-5" />
                </Button>
              </Link>

              <div className="mt-8 pt-8 border-t border-border grid grid-cols-1 sm:grid-cols-3 gap-4 text-left">
                <div className="flex items-start gap-2">
                  <Check className="w-4 h-4 text-success flex-shrink-0 mt-1" />
                  <span className="text-xs text-foreground-secondary font-mono">
                    Instant email delivery
                  </span>
                </div>
                <div className="flex items-start gap-2">
                  <Check className="w-4 h-4 text-success flex-shrink-0 mt-1" />
                  <span className="text-xs text-foreground-secondary font-mono">
                    All future updates
                  </span>
                </div>
                <div className="flex items-start gap-2">
                  <Check className="w-4 h-4 text-success flex-shrink-0 mt-1" />
                  <span className="text-xs text-foreground-secondary font-mono">
                    Use on unlimited rigs
                  </span>
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        {/* LICENSE VALIDATOR */}
        <section
          id="validator"
          className="py-20 lg:py-28 bg-background-secondary relative scroll-mt-24"
        >
          <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-10">
              <motion.span
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="inline-block font-mono text-xs uppercase tracking-widest text-cyan mb-4"
              >
                Already purchased?
              </motion.span>
              <motion.h2
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 }}
                className="text-3xl sm:text-4xl font-mono font-bold text-foreground mb-4"
              >
                Validate <span className="text-cyan">License</span>
              </motion.h2>
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
                className="text-foreground-secondary"
              >
                Confirm a key is active and tied to your email.
              </motion.p>
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="bg-gradient-to-br from-background-tertiary to-background-secondary border border-border rounded-2xl p-8"
            >
              <form
                onSubmit={handleValidate}
                className="flex flex-col sm:flex-row gap-3"
              >
                <div className="flex-1">
                  <Input
                    type="text"
                    placeholder="CLLDR-XXXX-XXXX-XXXX-XXXX"
                    value={licenseKey}
                    onChange={(e) =>
                      setLicenseKey(e.target.value.toUpperCase())
                    }
                    icon={<KeyRound className="w-4 h-4" />}
                    spellCheck={false}
                    autoComplete="off"
                    aria-label="License key"
                  />
                </div>
                <Button
                  type="submit"
                  variant="primary"
                  isLoading={validation.kind === "loading"}
                  disabled={validation.kind === "loading"}
                >
                  {validation.kind === "loading" ? "Checking" : "Validate"}
                </Button>
              </form>

              <div className="mt-6 min-h-[60px]">
                {validation.kind === "valid" && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-start gap-3 p-4 rounded-lg bg-success/10 border border-success/30"
                  >
                    <CheckCircle2 className="w-5 h-5 text-success flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-mono text-sm font-semibold text-success">
                        License is active
                      </p>
                      {validation.email && (
                        <p className="text-xs text-foreground-secondary mt-1 font-mono">
                          Registered to: {validation.email}
                        </p>
                      )}
                    </div>
                  </motion.div>
                )}

                {validation.kind === "invalid" && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-start gap-3 p-4 rounded-lg bg-error/10 border border-error/30"
                  >
                    <XCircle className="w-5 h-5 text-error flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-mono text-sm font-semibold text-error">
                        Invalid or unknown key
                      </p>
                      <p className="text-xs text-foreground-secondary mt-1">
                        Double-check the key from your purchase email.
                      </p>
                    </div>
                  </motion.div>
                )}

                {validation.kind === "error" && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-start gap-3 p-4 rounded-lg bg-amber/10 border border-amber/30"
                  >
                    <XCircle className="w-5 h-5 text-amber flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-mono text-sm font-semibold text-amber">
                        {validation.message}
                      </p>
                    </div>
                  </motion.div>
                )}

                {validation.kind === "loading" && (
                  <div className="flex items-center justify-center gap-2 p-4 text-foreground-secondary">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="font-mono text-sm">
                      Contacting license server
                    </span>
                  </div>
                )}
              </div>
            </motion.div>
          </div>
        </section>
      </main>
      <Footer />
    </>
  );
}
