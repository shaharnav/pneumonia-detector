import type { Metadata } from "next";
import { Inter } from "next/font/google"; // Use Inter for clinical aesthetic
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Pneumonia Detector",
  description: "Clinical grade pneumonia detection system",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} antialiased`}>
        {children}
      </body>
    </html>
  );
}
