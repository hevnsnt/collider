"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowRight, Zap, Users, Database } from "lucide-react";
import { Button } from "@/components/ui";
import { MatrixRain, GlitchText, AnimatedCounter, TypeWriter } from "@/components/effects";

// Mock data - would come from API
const stats = {
  hashrate: 847_000_000_000, // 847 GK/s
  workers: 1247,
  dps: 4_200_000_000, // 4.2B DPs
};

export function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-hero" />
      <MatrixRain opacity={0.08} />
      <div className="absolute inset-0 grid-bg opacity-50" />

      {/* Radial gradient overlay */}
      <div
        className="absolute inset-0"
        style={{
          background: "radial-gradient(ellipse at center, transparent 0%, rgba(10,10,10,0.8) 70%)"
        }}
      />

      {/* Scan line effect */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div
          className="absolute w-full h-px bg-cyan/20 animate-scan-line"
          style={{ top: "-100%" }}
        />
      </div>

      {/* Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 lg:py-32">
        <div className="text-center">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan/5 border border-cyan/20 mb-8"
          >
            <span className="status-online" />
            <span className="font-mono text-xs text-cyan uppercase tracking-wider">
              Pool Active - Puzzle #135
            </span>
          </motion.div>

          {/* Main Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-4xl sm:text-5xl lg:text-7xl font-mono font-bold tracking-tight mb-6"
          >
            <GlitchText
              text="COLLISION"
              className="text-foreground"
              glitchOnHover
              as="span"
            />
            <br />
            <span className="text-cyan text-glow-cyan">PROTOCOL</span>
          </motion.h1>

          {/* Subheadline */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="text-lg sm:text-xl text-foreground-secondary max-w-2xl mx-auto mb-4"
          >
            Distributed GPU computing for Bitcoin puzzle solving.
            <br className="hidden sm:block" />
            Join thousands of GPUs working together for collective rewards.
          </motion.p>

          {/* Typewriter effect */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="h-8 mb-8"
          >
            <TypeWriter
              texts={[
                "Solve Bitcoin puzzles together",
                "Earn proportional rewards",
                "K=1.15 - SOTA Kangaroo algorithm",
                "13.5 BTC waiting to be solved",
              ]}
              className="text-amber font-mono"
              speed={80}
              pauseDuration={3000}
            />
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16"
          >
            <Link href="/download">
              <Button size="lg">
                Join the Pool
                <ArrowRight className="w-5 h-5" />
              </Button>
            </Link>
            <Link href="/docs">
              <Button variant="secondary" size="lg">Learn More</Button>
            </Link>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-3xl mx-auto"
          >
            <StatCard
              icon={<Zap className="w-5 h-5" />}
              value={stats.hashrate}
              label="Total Hashrate"
              formatter={(v) => (v / 1e9).toFixed(0) + " GK/s"}
              delay={0.6}
            />
            <StatCard
              icon={<Users className="w-5 h-5" />}
              value={stats.workers}
              label="Active Workers"
              formatter={(v) => v.toLocaleString()}
              delay={0.7}
            />
            <StatCard
              icon={<Database className="w-5 h-5" />}
              value={stats.dps}
              label="DPs Collected"
              formatter={(v) => (v / 1e9).toFixed(1) + "B"}
              delay={0.8}
            />
          </motion.div>
        </div>
      </div>

      {/* Bottom gradient fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent" />
    </section>
  );
}

function StatCard({
  icon,
  value,
  label,
  formatter,
  delay,
}: {
  icon: React.ReactNode;
  value: number;
  label: string;
  formatter: (v: number) => string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className="relative group"
    >
      <div className="stat-card p-6 text-center">
        <div className="flex items-center justify-center gap-2 text-cyan mb-2">
          {icon}
        </div>
        <div className="text-2xl sm:text-3xl font-mono font-bold text-foreground mb-1 data-display">
          <AnimatedCounter
            value={value}
            formatFn={formatter}
            duration={2500}
          />
        </div>
        <div className="text-xs font-mono uppercase tracking-wider text-foreground-muted">
          {label}
        </div>
      </div>
    </motion.div>
  );
}
