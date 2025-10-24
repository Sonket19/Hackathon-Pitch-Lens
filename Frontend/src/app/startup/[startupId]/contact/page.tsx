import Header from "@/components/header";
import { StartupContactDirectory } from "@/components/startup-contact-directory";
import type { AnalysisData } from "@/lib/types";

export default async function StartupContactPage({
  params,
}: {
  params: { startupId: string };
}) {
  const { startupId } = params;
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

  let analysisData: AnalysisData | null = null;
  let error: string | null = null;

  if (!baseUrl) {
    error = "The API base URL is not configured.";
  } else {
    try {
      const response = await fetch(`${baseUrl}/deals/${startupId}`, {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error("Failed to load founder directory.");
      }
      analysisData = (await response.json()) as AnalysisData;
    } catch (err) {
      console.error("Failed to fetch founder contact details", err);
      error = err instanceof Error ? err.message : "Unexpected error while loading contacts.";
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <StartupContactDirectory
        analysisData={analysisData}
        startupId={startupId}
        initialError={error}
      />
    </div>
  );
}
