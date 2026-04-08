"use client";

import { motion } from "framer-motion";
import {
  Cpu,
  Scale,
  MonitorSmartphone,
  BarChart3,
  Shield,
  Zap,
  GitBranch,
  Trophy
} from "lucide-react";
import { Card } from "@/components/ui";

const features = [
  {
    icon: Cpu,
    title: "GPU Acceleration",
    description: "State-of-the-art CUDA kernels deliver 8+ GKeys/s on RTX 4090. Multi-GPU support scales linearly.",
    color: "cyan",
  },
  {
    icon: Scale,
    title: "Fair Rewards",
    description: "Proportional payout based on Distinguished Points contributed. No luck bonus - pure mathematics.",
    color: "amber",
  },
  {
    icon: MonitorSmartphone,
    title: "Multi-Platform",
    description: "Native binaries for Windows, Linux, and macOS (including Apple Silicon via Metal).",
    color: "cyan",
  },
  {
    icon: BarChart3,
    title: "Real-time Stats",
    description: "Live dashboard showing your hashrate, DPs, estimated share, and potential payout.",
    color: "amber",
  },
  {
    icon: Shield,
    title: "Secure Protocol",
    description: "JLP protocol with cryptographic verification. All submissions are validated on-chain.",
    color: "cyan",
  },
  {
    icon: Zap,
    title: "K=1.15 Efficiency",
    description: "SOTA Kangaroo implementation. 40-80% fewer operations than competitors at K=1.6-2.0.",
    color: "amber",
  },
  {
    icon: GitBranch,
    title: "Open Source",
    description: "Full source code on GitHub. Audit the algorithm, verify fairness, contribute improvements.",
    color: "cyan",
  },
  {
    icon: Trophy,
    title: "13.5 BTC Prize",
    description: "Puzzle #135 holds 13.5 BTC (~$1.4M). Join the hunt for one of crypto's biggest bounties.",
    color: "amber",
  },
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5 },
  },
};

export function Features() {
  return (
    <section className="py-24 lg:py-32 relative">
      {/* Background */}
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
            Why Choose Collision Protocol
          </motion.span>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-3xl sm:text-4xl lg:text-5xl font-mono font-bold text-foreground mb-6"
          >
            Built for <span className="text-cyan">Performance</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="text-lg text-foreground-secondary max-w-2xl mx-auto"
          >
            The most advanced GPU-accelerated Bitcoin puzzle solver, backed by cutting-edge
            cryptographic research and ruthless optimization.
          </motion.p>
        </div>

        {/* Features Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {features.map((feature) => (
            <motion.div key={feature.title} variants={itemVariants}>
              <Card variant="glow" className="h-full group">
                <div className="flex flex-col h-full">
                  <div
                    className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 transition-colors ${
                      feature.color === "cyan"
                        ? "bg-cyan/10 text-cyan group-hover:bg-cyan/20"
                        : "bg-amber/10 text-amber group-hover:bg-amber/20"
                    }`}
                  >
                    <feature.icon className="w-6 h-6" />
                  </div>
                  <h3 className="font-mono text-lg font-semibold text-foreground mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-foreground-secondary flex-grow">
                    {feature.description}
                  </p>
                </div>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
