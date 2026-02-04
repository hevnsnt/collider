import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(num: number): string {
  if (num >= 1e12) return (num / 1e12).toFixed(2) + "T";
  if (num >= 1e9) return (num / 1e9).toFixed(2) + "B";
  if (num >= 1e6) return (num / 1e6).toFixed(2) + "M";
  if (num >= 1e3) return (num / 1e3).toFixed(2) + "K";
  return num.toFixed(2);
}

export function formatHashrate(keysPerSecond: number): string {
  if (keysPerSecond >= 1e12) return (keysPerSecond / 1e12).toFixed(2) + " TK/s";
  if (keysPerSecond >= 1e9) return (keysPerSecond / 1e9).toFixed(2) + " GK/s";
  if (keysPerSecond >= 1e6) return (keysPerSecond / 1e6).toFixed(2) + " MK/s";
  if (keysPerSecond >= 1e3) return (keysPerSecond / 1e3).toFixed(2) + " KK/s";
  return keysPerSecond.toFixed(2) + " K/s";
}

export function formatBTC(amount: number): string {
  return amount.toFixed(8) + " BTC";
}

export function truncateAddress(address: string, chars = 6): string {
  return `${address.slice(0, chars)}...${address.slice(-chars)}`;
}
