import type { Metadata, Viewport } from "next";
import "./globals.css";
import Link from "next/link";
import { AuthProvider } from "../context/AuthContext";
import NavBar from "../components/NavBar";

export const metadata: Metadata = {
  title: "3D Layout - Blueprint Spatial Modeler",
  description: "Upload a blueprint image or use the guided wizard to generate an interactive 3D model.",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "3DLayout",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: "#000000",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">
        <AuthProvider>
          <NavBar />
          <main className="flex-1 pt-14">{children}</main>
        </AuthProvider>
      </body>
    </html>
  );
}
