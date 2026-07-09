import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";

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
      <body className="min-h-full flex flex-col">
        <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-3 bg-black/80 backdrop-blur-md border-b border-gray-900">
          <Link href="/" className="text-sm font-bold text-white tracking-tight">
            Blueprint Spatial Modeler
          </Link>
          <div className="flex items-center gap-4">
            <Link
              href="/login"
              className="text-xs font-semibold text-gray-400 hover:text-white transition-colors"
            >
              Sign In
            </Link>
          </div>
        </nav>
        <main className="flex-1 pt-14">{children}</main>
      </body>
    </html>
  );
}