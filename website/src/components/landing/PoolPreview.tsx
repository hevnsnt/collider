"use client";

import { motion } from "framer-motion";
import { TrendingUp, Activity, Target, Clock } from "lucide-react";
import { Card, Badge } from "@/components/ui";
import { AnimatedCounter } from "@/components/effects";

// Mock data - would come from real-time API
const poolStats = {
  hashrate: 847_000_000_000, // 847 GK/s
  workers: 1247,
  dps: 4_200_000_000,
  puzzleProgress: 0.0000042, // Very small percentage
  estimatedYears: 2.5,
};

const hashrateHistory = [
  { time: "00:00", value: 720 },
  { time: "04:00", value: 680 },
  { time: "08:00", value: 790 },
  { time: "12:00", value: 850 },
  { time: "16:00", value: 920 },
  { time: "20:00", value: 847 },
];

export function PoolPreview() {
  const maxValue = Math.max(...hashrateHistory.map((h) => h.value));

  return (
    <section className="py-24 lg:py-32 relative">
      <div className="absolute inset-0 grid-bg opacity-30" />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <motion.span
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="inline-block font-mono text-xs uppercase tracking-widest text-cyan mb-4"
          >
            Live Pool Statistics
          </motion.span>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-3xl sm:text-4xl lg:text-5xl font-mono font-bold text-foreground mb-6"
          >
            Real-time <span className="text-cyan">Pool Status</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="text-lg text-foreground-secondary max-w-2xl mx-auto"
          >
            Watch the collective computing power hunting for Puzzle #135.
            Every second brings us closer to the 13.5 BTC prize.
          </motion.p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Main Hashrate Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.3 }}
            className="lg:col-span-2"
          >
            <Card variant="stat" className="h-full">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <TrendingUp className="w-5 h-5 text-cyan" />
                    <span className="font-mono text-sm text-foreground-secondary uppercase tracking-wider">
                      Pool Hashrate
                    </span>
                  </div>
                  <div className="text-4xl sm:text-5xl font-mono font-bold text-foreground data-display">
                    <AnimatedCounter
                      value={poolStats.hashrate / 1e9}
                      formatFn={(v) => v.toFixed(0)}
                      duration={2000}
                    />
                    <span className="text-2xl ml-2 text-foreground-secondary">GK/s</span>
                  </div>
                </div>
                <Badge variant="success" pulse>
                  Live
                </Badge>
              </div>

              {/* Simple Bar Chart */}
              <div className="h-32 flex items-end gap-2">
                {hashrateHistory.map((point, index) => (
                  <motion.div
                    key={point.time}
                    initial={{ height: 0 }}
                    whileInView={{ height: `${(point.value / maxValue) * 100}%` }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.5 + index * 0.1, duration: 0.5 }}
                    className="flex-1 bg-gradient-to-t from-cyan/20 to-cyan/60 rounded-t relative group"
                  >
                    <div className="absolute -top-8 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <span className="font-mono text-xs text-cyan">{point.value}</span>
                    </div>
                    <div className="absolute -bottom-6 left-1/2 -translate-x-1/2">
                      <span className="font-mono text-2xs text-foreground-muted">{point.time}</span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </Card>
          </motion.div>

          {/* Side Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.4 }}
            className="space-y-6"
          >
            <Card variant="stat">
              <div className="flex items-center gap-3 mb-2">
                <Activity className="w-5 h-5 text-cyan" />
                <span className="font-mono text-sm text-foreground-secondary uppercase tracking-wider">
                  Active Workers
                </span>
              </div>
              <div className="text-3xl font-mono font-bold text-foreground data-display">
                <AnimatedCounter
                  value={poolStats.workers}
                  formatFn={(v) => v.toLocaleString()}
                  duration={2000}
                />
              </div>
            </Card>

            <Card variant="stat">
              <div className="flex items-center gap-3 mb-2">
                <Target className="w-5 h-5 text-amber" />
                <span className="font-mono text-sm text-foreground-secondary uppercase tracking-wider">
                  Current Target
                </span>
              </div>
              <div className="text-2xl font-mono font-bold text-foreground mb-2">
                Puzzle #135
              </div>
              <div className="text-sm text-amber font-mono">
                13.5 BTC Prize
              </div>
            </Card>

            <Card variant="stat">
              <div className="flex items-center gap-3 mb-2">
                <Clock className="w-5 h-5 text-cyan" />
                <span className="font-mono text-sm text-foreground-secondary uppercase tracking-wider">
                  Est. Time (Pool)
                </span>
              </div>
              <div className="text-3xl font-mono font-bold text-foreground">
                ~{poolStats.estimatedYears}
                <span className="text-lg ml-2 text-foreground-secondary">years</span>
              </div>
              <p className="text-xs text-foreground-muted mt-1">
                Based on current hashrate
              </p>
            </Card>
          </motion.div>
        </div>

        {/* Progress Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5 }}
        >
          <Card variant="outline" padding="lg">
            <div className="flex items-center justify-between mb-4">
              <span className="font-mono text-sm text-foreground-secondary">
                Puzzle #135 Progress
              </span>
              <span className="font-mono text-sm text-cyan">
                {(poolStats.puzzleProgress * 100).toFixed(6)}%
              </span>
            </div>
            <div className="h-3 bg-background rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                whileInView={{ width: `${Math.max(poolStats.puzzleProgress * 100, 0.5)}%` }}
                viewport={{ once: true }}
                transition={{ delay: 0.7, duration: 1 }}
                className="h-full bg-gradient-to-r from-cyan to-cyan-dim rounded-full relative"
              >
                <div className="absolute inset-0 bg-[linear-gradient(90deg,transparent,rgba(255,255,255,0.3),transparent)] animate-shimmer" />
              </motion.div>
            </div>
            <p className="text-xs text-foreground-muted mt-3 text-center">
              Based on Distinguished Points collected vs. expected total (~2^67.5 operations)
            </p>
          </Card>
        </motion.div>
      </div>
    </section>
  );
}
