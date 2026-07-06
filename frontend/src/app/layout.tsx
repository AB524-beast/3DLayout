import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "3D Layout - Blueprint Spatial Modeler",
  description: "Upload a blueprint image or use the guided wizard to generate an interactive 3D model.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}