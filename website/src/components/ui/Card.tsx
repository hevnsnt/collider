"use client";

import { forwardRef, HTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const cardVariants = cva(
  "border border-border rounded-lg transition-all duration-300",
  {
    variants: {
      variant: {
        default: [
          "bg-gradient-to-br from-background-tertiary to-background-secondary",
          "hover:border-border-hover hover:shadow-card-hover",
          "hover:-translate-y-0.5",
        ],
        glass: [
          "bg-background-secondary/80 backdrop-blur-md",
          "hover:bg-background-secondary/90",
        ],
        stat: [
          "bg-gradient-to-br from-background-tertiary to-background-secondary",
          "relative overflow-hidden",
          "before:absolute before:top-0 before:left-0 before:right-0 before:h-0.5",
          "before:bg-gradient-to-r before:from-transparent before:via-cyan before:to-transparent",
          "before:opacity-0 before:transition-opacity before:duration-300",
          "hover:before:opacity-100",
        ],
        outline: [
          "bg-transparent",
          "hover:bg-background-tertiary/50",
        ],
        glow: [
          "bg-gradient-to-br from-background-tertiary to-background-secondary",
          "hover:shadow-glow-cyan",
          "hover:border-cyan/30",
        ],
      },
      padding: {
        none: "p-0",
        sm: "p-4",
        md: "p-6",
        lg: "p-8",
      },
    },
    defaultVariants: {
      variant: "default",
      padding: "md",
    },
  }
);

export interface CardProps
  extends HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof cardVariants> {}

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant, padding, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(cardVariants({ variant, padding, className }))}
        {...props}
      />
    );
  }
);

Card.displayName = "Card";

// Card subcomponents
const CardHeader = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn("flex flex-col space-y-1.5", className)}
      {...props}
    />
  )
);
CardHeader.displayName = "CardHeader";

const CardTitle = forwardRef<HTMLHeadingElement, HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn(
        "font-mono text-lg font-semibold tracking-tight text-foreground",
        className
      )}
      {...props}
    />
  )
);
CardTitle.displayName = "CardTitle";

const CardDescription = forwardRef<HTMLParagraphElement, HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => (
    <p
      ref={ref}
      className={cn("text-sm text-foreground-secondary", className)}
      {...props}
    />
  )
);
CardDescription.displayName = "CardDescription";

const CardContent = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("", className)} {...props} />
  )
);
CardContent.displayName = "CardContent";

const CardFooter = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn("flex items-center pt-4", className)}
      {...props}
    />
  )
);
CardFooter.displayName = "CardFooter";

export { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter, cardVariants };
