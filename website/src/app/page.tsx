import { Navbar, Footer } from "@/components/layout";
import {
  Hero,
  Features,
  HowItWorks,
  PoolPreview,
  Comparison,
  CTA,
} from "@/components/landing";

export default function Home() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <Features />
        <HowItWorks />
        <PoolPreview />
        <Comparison />
        <CTA />
      </main>
      <Footer />
    </>
  );
}
