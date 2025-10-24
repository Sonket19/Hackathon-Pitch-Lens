'use client';

import Link from 'next/link';
import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { AnalysisData } from '@/lib/types';
import CompanyOverview from './company-overview';
import MarketAnalysis from './market-analysis';
import BusinessModel from './business-model';
import Financials from './financials';
import RiskAnalysis from './risk-analysis';
import Chatbot from './chatbot';
import {
  Briefcase,
  ShoppingCart,
  BarChart,
  Banknote,
  ShieldAlert,
  MessageCircle,
  SlidersHorizontal,
  Loader2,
  FileArchive,
  FileText,
  Video,
  Mic,
  Type,
} from 'lucide-react';
import { Button } from './ui/button';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Badge } from './ui/badge';
import { useToast } from '@/hooks/use-toast';

const DEFAULT_WEIGHTAGES = {
  teamStrength: 20,
  marketOpportunity: 20,
  traction: 20,
  claimCredibility: 25,
  financialHealth: 15,
} as const;

const SOURCE_FILE_CONFIG = {
  pitch_deck: {
    field: 'pitch_deck_url' as const,
    label: 'pitch_deck.pdf',
    endpoint: (startupId: string) => `/download_pitch_deck/${startupId}`,
    defaultName: (startupId: string) => `${startupId}-pitch-deck.pdf`,
    icon: FileText,
  },
  video_pitch: {
    field: 'video_pitch_deck_url' as const,
    label: 'founder_interview.mp4',
    endpoint: (startupId: string) => `/download_video_pitch/${startupId}`,
    defaultName: (startupId: string) => `${startupId}-video-pitch.mp4`,
    icon: Video,
  },
  audio_pitch: {
    field: 'audio_pitch_deck_url' as const,
    label: 'demo_walkthrough.mp3',
    endpoint: (startupId: string) => `/download_audio_pitch/${startupId}`,
    defaultName: (startupId: string) => `${startupId}-audio-pitch.mp3`,
    icon: Mic,
  },
  text_notes: {
    field: 'text_pitch_deck_url' as const,
    label: 'additional_notes.txt',
    endpoint: (startupId: string) => `/download_text_notes/${startupId}`,
    defaultName: (startupId: string) => `${startupId}-text-notes.txt`,
    icon: Type,
  },
} satisfies Record<
  'pitch_deck' | 'video_pitch' | 'audio_pitch' | 'text_notes',
  {
    field: keyof AnalysisData['raw_files'];
    label: string;
    endpoint: (startupId: string) => string;
    defaultName: (startupId: string) => string;
    icon: typeof FileText;
  }
>;

type WeightKey = keyof typeof DEFAULT_WEIGHTAGES;

type AnalysisDashboardProps = {
  analysisData: AnalysisData;
  startupId: string;
};

type SourceFileKey = keyof typeof SOURCE_FILE_CONFIG;

const formatWeightLabel = (key: WeightKey) =>
  key.replace(/([A-Z])/g, ' $1').replace(/^./, char => char.toUpperCase());

const NoDataComponent = ({ onGenerateClick }: { onGenerateClick: () => void }) => (
  <div className="rounded-lg border-2 border-dashed py-20 text-center">
    <h2 className="font-headline text-2xl font-semibold">Analysis data is not available.</h2>
    <p className="mt-2 text-muted-foreground">
      The analysis for this startup might still be in progress or has failed. You can generate a summary.
    </p>
    <Button onClick={onGenerateClick} className="mt-4">
      <SlidersHorizontal className="mr-2 h-4 w-4" />
      Generate Summary
    </Button>
  </div>
);

export default function AnalysisDashboard({ analysisData: initialAnalysisData, startupId }: AnalysisDashboardProps) {
  const [analysisData, setAnalysisData] = useState(initialAnalysisData);
  const [weights, setWeights] = useState(DEFAULT_WEIGHTAGES);
  const [isCustomizeDialogOpen, setIsCustomizeDialogOpen] = useState(false);
  const [isRecalculating, setIsRecalculating] = useState(false);
  const [downloadingFile, setDownloadingFile] = useState<SourceFileKey | null>(null);
  const { toast } = useToast();

  const memo = analysisData?.memo?.draft_v1;
  const rawFiles = (analysisData?.raw_files ?? {}) as Record<string, string | undefined>;
  const totalWeight = (Object.values(weights) as number[]).reduce((sum, value) => sum + value, 0);

  const handleWeightChange = (key: WeightKey, value: number[]) => {
    setWeights(prev => ({ ...prev, [key]: value[0] ?? prev[key] }));
  };

  const handleRecalculate = async () => {
    setIsRecalculating(true);

    const requestBody = {
      team_strength: weights.teamStrength,
      market_opportunity: weights.marketOpportunity,
      traction: weights.traction,
      claim_credibility: weights.claimCredibility,
      financial_health: weights.financialHealth,
    };

    try {
      const generateMemoResponse = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/generate_memo/${startupId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!generateMemoResponse.ok) {
        throw new Error('Failed to generate the new memo summary.');
      }

      const generateResult = await generateMemoResponse.json();
      const dealResponse = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/deals/${generateResult.deal_id}`);

      if (!dealResponse.ok) {
        throw new Error('Failed to fetch the updated analysis data.');
      }

      const updatedAnalysisData = (await dealResponse.json()) as AnalysisData;
      setAnalysisData(updatedAnalysisData);
      setIsCustomizeDialogOpen(false);

      toast({
        title: 'Summary Generated',
        description: 'The investment summary has been updated with your new weightages.',
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'An unexpected error occurred while regenerating the summary.';
      toast({
        variant: 'destructive',
        title: 'Update Failed',
        description: message,
      });
    } finally {
      setIsRecalculating(false);
    }
  };

  const handleDownloadSourceFile = async (key: SourceFileKey) => {
    setDownloadingFile(key);

    const config = SOURCE_FILE_CONFIG[key];
    const endpoint = config.endpoint(startupId);
    const defaultFilename = config.defaultName(startupId);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}${endpoint}`);

      if (!response.ok) {
        throw new Error(`Failed to download ${config.label}.`);
      }

      const blob = await response.blob();
      const contentDisposition = response.headers.get('content-disposition');
      let filename = defaultFilename;

      if (contentDisposition) {
        const match = contentDisposition.match(/filename="(.+)"/);
        if (match?.[1]) {
          filename = match[1];
        }
      }

      const url = window.URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.URL.revokeObjectURL(url);

      toast({
        title: 'Download Started',
        description: `Your download for ${filename} has started.`,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'An unexpected error occurred while downloading the file.';
      toast({
        variant: 'destructive',
        title: 'Download Failed',
        description: message,
      });
    } finally {
      setDownloadingFile(null);
    }
  };

  return (
    <div className="w-full animate-in fade-in-50 duration-500">
      <div className="mb-4 flex flex-wrap items-center justify-end gap-4">
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline">
              <FileArchive className="mr-2 h-4 w-4" /> Uploaded Data
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[525px]">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-3 font-headline text-2xl">
                <FileArchive className="h-7 w-7 text-primary" /> Uploaded Data Sources
              </DialogTitle>
              <DialogDescription>Download the original source files used for this analysis.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              {(Object.keys(SOURCE_FILE_CONFIG) as SourceFileKey[]).map(fileKey => {
                const config = SOURCE_FILE_CONFIG[fileKey];
                const url = rawFiles[config.field as string];
                if (!url) {
                  return null;
                }

                const Icon = config.icon;
                const isDownloading = downloadingFile === fileKey;

                return (
                  <div key={fileKey} className="flex items-center justify-between rounded-lg bg-secondary/50 p-3">
                    <div className="flex items-center gap-3">
                      <Icon className="h-6 w-6 text-muted-foreground" />
                      <span className="font-medium">{config.label}</span>
                    </div>
                    <Button size="sm" onClick={() => handleDownloadSourceFile(fileKey)} disabled={isDownloading}>
                      {isDownloading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Download'}
                    </Button>
                  </div>
                );
              })}
              {Object.values(SOURCE_FILE_CONFIG).every(config => !rawFiles[config.field as string]) ? (
                <p className="text-sm text-muted-foreground">No source files are available for download.</p>
              ) : null}
            </div>
          </DialogContent>
        </Dialog>

        <div className="flex flex-wrap items-center gap-3">
          <Button asChild variant="secondary">
            <Link href={`/startup/${startupId}/contact`}>Contact</Link>
          </Button>
          <Dialog open={isCustomizeDialogOpen} onOpenChange={setIsCustomizeDialogOpen}>
            <DialogTrigger asChild>
              <Button>
                <SlidersHorizontal className="mr-2 h-4 w-4" /> {memo ? 'Regenerate' : 'Generate'} Summary
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[625px]">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-3 font-headline text-2xl">
                  <SlidersHorizontal className="h-7 w-7 text-primary" /> Customize Score Weightage
                </DialogTitle>
                <DialogDescription>
                  Adjust the importance of each factor to recalculate the safety score. The total must be 100%.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-6 py-4">
                <div className="grid grid-cols-1 gap-x-8 gap-y-4 md:grid-cols-2">
                  {(Object.keys(weights) as WeightKey[]).map(key => (
                    <div key={key} className="grid gap-2">
                      <div className="flex items-center justify-between">
                        <Label htmlFor={key}>{formatWeightLabel(key)}</Label>
                        <span className="text-sm font-medium">{weights[key]}%</span>
                      </div>
                      <Slider
                        id={key}
                        value={[weights[key]]}
                        onValueChange={value => handleWeightChange(key, value)}
                        max={100}
                        step={5}
                      />
                    </div>
                  ))}
                </div>
                <div className="flex items-center justify-end gap-2 text-sm">
                  <Label>Total Weight:</Label>
                  <Badge variant={totalWeight === 100 ? 'default' : 'destructive'}>{totalWeight}%</Badge>
                </div>
              </div>
              <DialogFooter>
                <DialogClose asChild>
                  <Button variant="ghost">Close</Button>
                </DialogClose>
                <Button onClick={handleRecalculate} disabled={totalWeight !== 100 || isRecalculating}>
                  {isRecalculating ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ShieldAlert className="mr-2 h-4 w-4" />}
                  {isRecalculating ? 'Recalculating...' : 'Generate Summary'}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

        </div>
      </div>

      <Tabs defaultValue="overview">
        <TabsList className="mb-6 grid h-auto w-full grid-cols-2 md:grid-cols-6">
          <TabsTrigger value="overview" className="h-12">
            <Briefcase className="mr-2" /> Overview
          </TabsTrigger>
          <TabsTrigger value="market" className="h-12">
            <ShoppingCart className="mr-2" /> Market
          </TabsTrigger>
          <TabsTrigger value="model" className="h-12">
            <BarChart className="mr-2" /> Business Model
          </TabsTrigger>
          <TabsTrigger value="financials" className="h-12">
            <Banknote className="mr-2" /> Financials
          </TabsTrigger>
          <TabsTrigger value="risks" className="h-12">
            <ShieldAlert className="mr-2" /> Risks
          </TabsTrigger>
          <TabsTrigger value="chatbot" className="h-12">
            <MessageCircle className="mr-2" /> Chatbot
          </TabsTrigger>
        </TabsList>
        <TabsContent value="overview">
          {memo ? <CompanyOverview data={memo.company_overview} /> : <NoDataComponent onGenerateClick={() => setIsCustomizeDialogOpen(true)} />}
        </TabsContent>
        <TabsContent value="market">
          {memo ? <MarketAnalysis data={memo.market_analysis} /> : <NoDataComponent onGenerateClick={() => setIsCustomizeDialogOpen(true)} />}
        </TabsContent>
        <TabsContent value="model">
          {memo ? <BusinessModel data={memo.business_model} dealId={startupId} /> : <NoDataComponent onGenerateClick={() => setIsCustomizeDialogOpen(true)} />}
        </TabsContent>
        <TabsContent value="financials">
          {memo ? <Financials data={memo.financials} claims={memo.claims_analysis} /> : <NoDataComponent onGenerateClick={() => setIsCustomizeDialogOpen(true)} />}
        </TabsContent>
        <TabsContent value="risks">
          {memo ? (
            <RiskAnalysis riskMetrics={memo.risk_metrics} conclusion={memo.conclusion} isRecalculating={isRecalculating} />
          ) : (
            <NoDataComponent onGenerateClick={() => setIsCustomizeDialogOpen(true)} />
          )}
        </TabsContent>
        <TabsContent value="chatbot">
          {memo ? <Chatbot analysisData={analysisData} /> : <NoDataComponent onGenerateClick={() => setIsCustomizeDialogOpen(true)} />}
        </TabsContent>
      </Tabs>
    </div>
  );
}
