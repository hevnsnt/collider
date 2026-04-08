"use client";

import { motion } from "framer-motion";
import { Download, Plug, Coins, ArrowRight } from "lucide-react";

const steps = [
  {
    number: "01",
    icon: Download,
    title: "Download Client",
    description: "Get theCollider for your platform. Available for Windows, Linux, and macOS with full GPU support.",
    code: "curl -L https://colliderprotocol.com/download | sh",
  },
  {
    number: "02",
    icon: Plug,
    title: "Connect to Pool",
    description: "Launch the client with your worker name. Automatic GPU detection and optimal configuration.",
    code: "./collider --pool jlp://pool.colliderprotocol.com:17403 --worker YourName",
  },
  {
    number: "03",
    icon: Coins,
    title: "Earn Rewards",
    description: "Your GPUs contribute Distinguished Points. When the puzzle is solved, rewards are distributed proportionally.",
    code: "// Your share = (Your DPs / Total DPs) * Prize",
  },
];

export function HowItWorks() {
  return (
    <section className="py-24 lg:py-32 bg-background-secondary relative overflow-hidden">
      {/* Background decoration */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-cyan/5 rounded-full blur-3xl" />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <motion.span
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="inline-block font-mono text-xs uppercase tracking-widest text-cyan mb-4"
          >
            Getting Started
          </motion.span>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-3xl sm:text-4xl lg:text-5xl font-mono font-bold text-foreground mb-6"
          >
            Three Steps to <span className="text-cyan">Start Mining</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="text-lg text-foreground-secondary max-w-2xl mx-auto"
          >
            From zero to contributing in under 5 minutes. No complex setup, no configuration files.
          </motion.p>
        </div>

        {/* Steps */}
        <div className="relative">
          {/* Connection line */}
          <div className="hidden lg:block absolute top-1/2 left-0 right-0 h-px bg-gradient-to-r from-transparent via-border to-transparent -translate-y-1/2" />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 lg:gap-12">
            {steps.map((step, index) => (
              <motion.div
                key={step.number}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.2 }}
                className="relative"
              >
                {/* Step Card */}
                <div className="bg-background border border-border rounded-lg p-8 relative group hover:border-cyan/30 transition-colors">
                  {/* Step number badge */}
                  <div className="absolute -top-4 left-8 bg-cyan text-background font-mono text-sm font-bold px-3 py-1 rounded">
                    {step.number}
                  </div>

                  {/* Icon */}
                  <div className="w-14 h-14 rounded-lg bg-cyan/10 flex items-center justify-center mb-6 group-hover:bg-cyan/20 transition-colors">
                    <step.icon className="w-7 h-7 text-cyan" />
                  </div>

                  {/* Content */}
                  <h3 className="font-mono text-xl font-semibold text-foreground mb-3">
                    {step.title}
                  </h3>
                  <p className="text-foreground-secondary mb-6">
                    {step.description}
                  </p>

                  {/* Code block */}
                  <div className="bg-background-secondary rounded-md p-4 border border-border overflow-x-auto">
                    <code className="font-mono text-xs text-cyan whitespace-nowrap">
                      {step.code}
                    </code>
                  </div>
                </div>

                {/* Arrow connector (visible on lg+) */}
                {index < steps.length - 1 && (
                  <div className="hidden lg:flex absolute top-1/2 -right-6 w-12 items-center justify-center -translate-y-1/2 z-10">
                    <ArrowRight className="w-6 h-6 text-cyan" />
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>

        {/* Bottom CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.6 }}
          className="text-center mt-16"
        >
          <p className="text-foreground-secondary mb-4">
            Ready to start? Check out our detailed documentation.
          </p>
          <a
            href="/docs/getting-started"
            className="inline-flex items-center gap-2 font-mono text-cyan hover:text-cyan-dim transition-colors"
          >
            Read the full guide
            <ArrowRight className="w-4 h-4" />
          </a>
        </motion.div>
      </div>
    </section>
  );
}
