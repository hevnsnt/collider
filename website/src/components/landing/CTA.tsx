"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowRight, Github, Zap } from "lucide-react";
import { Button } from "@/components/ui";

export function CTA() {
  return (
    <section className="py-24 lg:py-32 relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 bg-gradient-to-br from-cyan/5 via-background to-amber/5" />
      <div className="absolute inset-0 grid-bg opacity-20" />

      {/* Glow orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan/10 rounded-full blur-3xl" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-amber/10 rounded-full blur-3xl" />

      <div className="relative z-10 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="bg-gradient-to-br from-background-tertiary to-background-secondary border border-border rounded-2xl p-8 sm:p-12 lg:p-16"
        >
          {/* Icon */}
          <motion.div
            initial={{ scale: 0 }}
            whileInView={{ scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2, type: "spring" }}
            className="w-20 h-20 mx-auto mb-8 rounded-2xl bg-cyan/10 border border-cyan/20 flex items-center justify-center"
          >
            <Zap className="w-10 h-10 text-cyan" />
          </motion.div>

          {/* Headline */}
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.3 }}
            className="text-3xl sm:text-4xl lg:text-5xl font-mono font-bold text-foreground mb-6"
          >
            Ready to <span className="text-cyan">Join the Hunt</span>&#63;
          </motion.h2>

          {/* Description */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.4 }}
            className="text-lg text-foreground-secondary mb-8 max-w-2xl mx-auto"
          >
            13.5 BTC is waiting. Every GPU matters. Download theCollider, connect to the pool,
            and start contributing to one of crypto&apos;s most challenging puzzles.
          </motion.p>

          {/* Prize highlight */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.5 }}
            className="inline-flex items-center gap-4 bg-amber/10 border border-amber/20 rounded-lg px-6 py-4 mb-10"
          >
            <span className="text-4xl font-mono font-bold text-amber">13.5 BTC</span>
            <div className="text-left">
              <p className="text-sm text-foreground-secondary">Current Prize</p>
              <p className="text-xs text-foreground-muted">Puzzle #135</p>
            </div>
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.6 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link href="/download">
              <Button size="lg">
                Download Client
                <ArrowRight className="w-5 h-5" />
              </Button>
            </Link>
            <Link href="https://github.com/hevnsnt/theCollider" target="_blank" rel="noopener noreferrer">
              <Button variant="secondary" size="lg">
                <Github className="w-5 h-5" />
                View on GitHub
              </Button>
            </Link>
          </motion.div>

          {/* Trust indicators */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.8 }}
            className="mt-10 pt-8 border-t border-border flex flex-wrap items-center justify-center gap-6 text-sm text-foreground-muted"
          >
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-success" />
              5% Pool Fee
            </span>
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-success" />
              Transparent Payouts
            </span>
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-success" />
              Open Source
            </span>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
