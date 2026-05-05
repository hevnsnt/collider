"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { CheckCircle2, Mail, Terminal, BookOpen, ArrowRight } from "lucide-react";

import { Navbar, Footer } from "@/components/layout";
import { Button } from "@/components/ui";

export default function SuccessPage() {
  return (
    <>
      <Navbar />
      <main className="bg-background min-h-screen">
        <section className="relative pt-32 pb-20 lg:pt-40 lg:pb-28 overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-cyan/5 via-background to-amber/5" />
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan/10 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-amber/10 rounded-full blur-3xl" />

          <div className="relative z-10 max-w-2xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gradient-to-br from-background-tertiary to-background-secondary border border-border rounded-2xl p-8 sm:p-12 text-center"
            >
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
                className="w-20 h-20 mx-auto mb-8 rounded-2xl bg-success/10 border border-success/30 flex items-center justify-center"
              >
                <CheckCircle2 className="w-10 h-10 text-success" />
              </motion.div>

              <motion.h1
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="text-3xl sm:text-4xl lg:text-5xl font-mono font-bold text-foreground mb-4"
              >
                Purchase <span className="text-cyan">Complete</span>
              </motion.h1>

              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="text-lg text-foreground-secondary mb-10"
              >
                Welcome to theCollider Pro.
              </motion.p>

              {/* Email step */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-background border border-border rounded-xl p-6 mb-4 text-left"
              >
                <div className="flex items-start gap-4">
                  <div className="w-10 h-10 rounded-lg bg-cyan/10 border border-cyan/20 flex items-center justify-center flex-shrink-0">
                    <Mail className="w-5 h-5 text-cyan" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h2 className="font-mono text-sm uppercase tracking-wider text-cyan mb-2">
                      Step 1 &middot; Check your email
                    </h2>
                    <p className="text-sm text-foreground-secondary">
                      We just sent your{" "}
                      <span className="font-mono text-foreground">
                        CLLDR-XXXX
                      </span>{" "}
                      license key. If it does not arrive in 5 minutes, check
                      spam or contact{" "}
                      <a
                        href="mailto:hevnsnt@gmail.com"
                        className="text-cyan hover:text-cyan-dim transition-colors"
                      >
                        support
                      </a>
                      .
                    </p>
                  </div>
                </div>
              </motion.div>

              {/* Activation step */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="bg-background border border-border rounded-xl p-6 mb-8 text-left"
              >
                <div className="flex items-start gap-4">
                  <div className="w-10 h-10 rounded-lg bg-amber/10 border border-amber/20 flex items-center justify-center flex-shrink-0">
                    <Terminal className="w-5 h-5 text-amber" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h2 className="font-mono text-sm uppercase tracking-wider text-amber mb-2">
                      Step 2 &middot; Activate
                    </h2>
                    <p className="text-sm text-foreground-secondary mb-3">
                      Run this command in the directory where{" "}
                      <code className="font-mono text-foreground">
                        collider
                      </code>{" "}
                      lives:
                    </p>
                    <pre className="bg-background-elevated border border-border rounded-md p-3 overflow-x-auto">
                      <code className="font-mono text-xs text-amber">
                        ./collider --activate YOUR_KEY
                      </code>
                    </pre>
                  </div>
                </div>
              </motion.div>

              {/* CTAs */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 }}
                className="flex flex-col sm:flex-row items-center justify-center gap-3"
              >
                <Link href="/dashboard">
                  <Button size="md">
                    <ArrowRight className="w-4 h-4" />
                    Go to Dashboard
                  </Button>
                </Link>
                <Link href="/docs">
                  <Button variant="secondary" size="md">
                    <BookOpen className="w-4 h-4" />
                    Read the Docs
                  </Button>
                </Link>
              </motion.div>
            </motion.div>
          </div>
        </section>
      </main>
      <Footer />
    </>
  );
}
