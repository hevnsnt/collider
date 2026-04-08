"use client";

import { forwardRef, HTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center font-mono text-xs uppercase tracking-wider transition-colors",
  {
    variants: {
      variant: {
        default: "bg-background-elevated text-foreground-secondary border border-border",
        cyan: "bg-cyan/10 text-cyan border border-cyan/30",
        amber: "bg-amber/10 text-amber border border-amber/30",
        success: "bg-success/10 text-success border border-success/30",
        error: "bg-error/10 text-error border border-error/30",
        outline: "bg-transparent text-foreground-secondary border border-border",
      },
      size: {
        sm: "px-2 py-0.5 rounded text-2xs",
        md: "px-2.5 py-1 rounded-md",
        lg: "px-3 py-1.5 rounded-md text-sm",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "md",
    },
  }
);

export interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {
  pulse?: boolean;
}

const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant, size, pulse, children, ...props }, ref) => {
    return (
      <span
        ref={ref}
        className={cn(badgeVariants({ variant, size }), className)}
        {...props}
      >
        {pulse && (
          <span className="relative flex h-2 w-2 mr-1.5">
            <span
              className={cn(
                "animate-ping absolute inline-flex h-full w-full rounded-full opacity-75",
                variant === "success" && "bg-success",
                variant === "error" && "bg-error",
                variant === "amber" && "bg-amber",
                variant === "cyan" && "bg-cyan",
                !variant && "bg-foreground-secondary"
              )}
            />
            <span
              className={cn(
                "relative inline-flex rounded-full h-2 w-2",
                variant === "success" && "bg-success",
                variant === "error" && "bg-error",
                variant === "amber" && "bg-amber",
                variant === "cyan" && "bg-cyan",
                !variant && "bg-foreground-secondary"
              )}
            />
          </span>
        )}
        {children}
      </span>
    );
  }
);

Badge.displayName = "Badge";

export { Badge, badgeVariants };
