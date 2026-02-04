"use client";

import { motion } from "framer-motion";
import { Check, X, Minus } from "lucide-react";

const tools = [
  "theCollider",
  "BitCrack",
  "VanitySearch",
  "KeyHunt",
  "JLP Kangaroo",
  "RCKangaroo",
];

const features = [
  {
    name: "Kangaroo K Value",
    values: ["1.15", "N/A", "N/A", "1.6-2.0", "1.6-2.0", "1.15"],
    highlight: true,
  },
  {
    name: "Brain Wallet Mode",
    values: [true, false, false, "Limited", false, false],
  },
  {
    name: "Bloom Filter Integration",
    values: [true, false, false, false, false, false],
    highlight: true,
  },
  {
    name: "Multi-GPU Support",
    values: [true, true, true, true, true, true],
  },
  {
    name: "macOS Support",
    values: [true, false, false, "Limited", false, false],
    highlight: true,
  },
  {
    name: "PCFG Generation",
    values: [true, false, false, false, false, false],
  },
  {
    name: "Rule Engine",
    values: [true, false, false, false, false, false],
  },
  {
    name: "Interactive Mode",
    values: [true, false, false, false, false, false],
  },
  {
    name: "Auto DP-Bits Selection",
    values: [true, false, false, false, false, false],
    highlight: true,
  },
  {
    name: "Opportunistic Scanning",
    values: [true, false, false, false, false, false],
    highlight: true,
  },
];

function CellValue({ value }: { value: boolean | string }) {
  if (value === true) {
    return <Check className="w-5 h-5 text-success" />;
  }
  if (value === false) {
    return <X className="w-5 h-5 text-error/50" />;
  }
  if (value === "Limited") {
    return <span className="text-amber text-xs font-mono">Limited</span>;
  }
  if (value === "N/A") {
    return <Minus className="w-4 h-4 text-foreground-muted" />;
  }
  return <span className="font-mono text-sm text-foreground">{value}</span>;
}

export function Comparison() {
  return (
    <section className="py-24 lg:py-32 bg-background-secondary relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <motion.span
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="inline-block font-mono text-xs uppercase tracking-widest text-cyan mb-4"
          >
            Competitor Analysis
          </motion.span>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-3xl sm:text-4xl lg:text-5xl font-mono font-bold text-foreground mb-6"
          >
            Why <span className="text-cyan">theCollider</span> Wins
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="text-lg text-foreground-secondary max-w-2xl mx-auto"
          >
            A comprehensive toolkit that outperforms every alternative.
            The K=1.15 advantage alone saves 40-80% of compute time.
          </motion.p>
        </div>

        {/* Comparison Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3 }}
          className="overflow-x-auto"
        >
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-4 px-4 font-mono text-sm text-foreground-secondary uppercase tracking-wider">
                  Feature
                </th>
                {tools.map((tool, index) => (
                  <th
                    key={tool}
                    className={`text-center py-4 px-4 font-mono text-sm uppercase tracking-wider ${
                      index === 0
                        ? "text-cyan bg-cyan/5"
                        : "text-foreground-secondary"
                    }`}
                  >
                    {tool}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {features.map((feature) => (
                <tr
                  key={feature.name}
                  className={`border-b border-border/50 ${
                    feature.highlight ? "bg-cyan/5" : ""
                  }`}
                >
                  <td className="py-4 px-4 text-sm text-foreground">
                    {feature.name}
                    {feature.highlight && (
                      <span className="ml-2 text-cyan text-xs">*</span>
                    )}
                  </td>
                  {feature.values.map((value, colIndex) => (
                    <td
                      key={colIndex}
                      className={`text-center py-4 px-4 ${
                        colIndex === 0 ? "bg-cyan/5" : ""
                      }`}
                    >
                      <div className="flex items-center justify-center">
                        <CellValue value={value} />
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>

        {/* Key Advantage Callout */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5 }}
          className="mt-12 p-6 bg-background border border-cyan/20 rounded-lg"
        >
          <div className="flex flex-col md:flex-row items-start md:items-center gap-4">
            <div className="flex-shrink-0 w-16 h-16 rounded-lg bg-cyan/10 flex items-center justify-center">
              <span className="font-mono text-2xl font-bold text-cyan">K</span>
            </div>
            <div>
              <h3 className="font-mono text-lg font-semibold text-foreground mb-2">
                The K=1.15 Advantage
              </h3>
              <p className="text-foreground-secondary">
                For a 135-bit puzzle, classic Kangaroo (K=2.0) needs ~2^68.5 operations.
                With K=1.15, theCollider needs ~2^67.3 operations: a <strong className="text-cyan">2.3x reduction</strong>.
                That translates to weeks of saved compute time.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
