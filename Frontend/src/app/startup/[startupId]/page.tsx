import StartupAnalysisClient from '@/components/startup-analysis-client';

interface StartupPageParams {
  startupId: string;
}

export default async function StartupPage({
  params,
}: {
  params: StartupPageParams | Promise<StartupPageParams>;
}) {
  const resolvedParams = await Promise.resolve(params);

  return <StartupAnalysisClient startupId={resolvedParams.startupId} />;
}
