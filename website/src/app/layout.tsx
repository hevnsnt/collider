import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "Collision Protocol | Distributed GPU Computing for Bitcoin Puzzles",
  description: "Join thousands of GPUs working together to solve Bitcoin puzzles. theCollider delivers state-of-the-art Kangaroo algorithm performance with K=1.15 efficiency.",
  keywords: ["Bitcoin", "puzzle", "GPU", "mining", "Kangaroo", "cryptocurrency", "distributed computing"],
  authors: [{ name: "Collision Protocol" }],
  openGraph: {
    title: "Collision Protocol | Distributed GPU Computing for Bitcoin Puzzles",
    description: "Join thousands of GPUs working together to solve Bitcoin puzzles.",
    type: "website",
    url: "https://colliderprotocol.com",
  },
  twitter: {
    card: "summary_large_image",
    title: "Collision Protocol",
    description: "Distributed GPU Computing for Bitcoin Puzzles",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
      </body>
    </html>
  );
}
