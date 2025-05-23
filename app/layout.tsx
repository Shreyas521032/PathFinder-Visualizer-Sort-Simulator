// app/layout.tsx
import { Inter } from 'next/font/google';
import { Toaster } from 'react-hot-toast';
import Navigation from '@/components/Navigation';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata = {
  title: 'PathFinder Visualizer | Algorithm Visualization on Real Maps',
  description: 'Interactive pathfinding and sorting algorithm visualizer with real-world maps. Built with Next.js, TypeScript, and Mapbox.',
  keywords: 'pathfinding, algorithms, visualization, sorting, A*, Dijkstra, BFS, DFS, maps, Mapbox',
  authors: [{ name: 'Shreyas Kasture' }],
  creator: 'Shreyas Kasture',
  openGraph: {
    title: 'PathFinder Visualizer',
    description: 'Interactive algorithm visualization on real maps',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'PathFinder Visualizer',
    description: 'Interactive algorithm visualization on real maps',
  },
  viewport: 'width=device-width, initial-scale=1',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#111827' },
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#3B82F6" />
      </head>
      <body className={`${inter.className} bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-white transition-colors duration-300`}>
        <div className="min-h-screen flex flex-col">
          <Navigation />
          <main className="flex-1">
            {children}
          </main>
          <footer className="bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 py-6 mt-auto">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="text-center text-sm text-gray-500 dark:text-gray-400">
                <p>
                  Developed with ❤️ by{' '}
                  <span className="font-semibold text-gray-700 dark:text-gray-300">
                    Shreyas Kasture
                  </span>
                </p>
                <p className="mt-1">
                  Open source project for learning algorithms and data structures
                </p>
              </div>
            </div>
          </footer>
        </div>
        <Toaster
          position="bottom-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: 'var(--toast-bg)',
              color: 'var(--toast-color)',
              border: '1px solid var(--toast-border)',
            },
            success: {
              iconTheme: {
                primary: '#10B981',
                secondary: '#FFFFFF',
              },
            },
            error: {
              iconTheme: {
                primary: '#EF4444',
                secondary: '#FFFFFF',
              },
            },
          }}
        />
      </body>
    </html>
  );
}
