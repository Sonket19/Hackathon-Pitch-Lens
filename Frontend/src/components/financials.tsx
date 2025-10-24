
'use client';

import { useState, type ReactNode } from 'react';
import type { Financials as FinancialsType, ClaimsAnalysis as ClaimsAnalysisType } from '@/lib/types';
import { generateFinancialMetricsDashboard } from '@/ai/flows/financial-metrics-dashboard';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { BarChart2, Briefcase, Calendar, Target, HelpCircle, GitBranch, PiggyBank, Sparkles, Loader2, Brain } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import {
  ResponsiveContainer,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  BarChart,
  Bar,
  Legend,
} from 'recharts';

type MetricCardProps = {
  title: string;
  value?: string | number | null;
  icon: ReactNode;
  tooltip?: string;
  formatAsCurrency?: boolean;
};

type ChartDatum = {
  name: string;
  value: number;
  display?: string;
  fullLabel?: string;
};

const isValueUnavailable = (value?: string | null) => {
  if (!value) {
    return true;
  }

  const normalized = value.trim().toLowerCase();
  if (!normalized) {
    return true;
  }

  return ['n/a', 'na', 'not available', 'none', 'unknown', 'tbd', '-', '—', 'pending', 'not applicable'].some((token) =>
    normalized.includes(token),
  );
};

const parseFinancialNumber = (value?: string | null): number | null => {
  if (!value || isValueUnavailable(value)) {
    return null;
  }

  const normalized = value.toString().trim().toLowerCase();
  const cleaned = normalized.replace(/,/g, '');
  const match = cleaned.match(/-?\d+(\.\d+)?/);

  if (!match) {
    return null;
  }

  let numeric = parseFloat(match[0]);
  if (Number.isNaN(numeric)) {
    return null;
  }

  const startIndex = match.index ?? 0;
  const before = cleaned.slice(0, startIndex);
  const after = cleaned.slice(startIndex + match[0].length);
  const tokens = `${before} ${after}`
    .match(/[a-z]+/g)
    ?.map((token) => token.toLowerCase()) ?? [];

  const multipliers: Record<string, number> = {
    trillion: 1_000_000_000_000,
    tn: 1_000_000_000_000,
    t: 1_000_000_000_000,
    billion: 1_000_000_000,
    bn: 1_000_000_000,
    b: 1_000_000_000,
    million: 1_000_000,
    mn: 1_000_000,
    mm: 1_000_000,
    thousand: 1_000,
    k: 1_000,
    crore: 10_000_000,
    crores: 10_000_000,
    cr: 10_000_000,
    lakh: 100_000,
    lakhs: 100_000,
    lac: 100_000,
  };

  for (const token of tokens) {
    if (multipliers[token]) {
      return numeric * multipliers[token];
    }
  }

  const suffixWord = after.trim().match(/^[a-z]+/i)?.[0]?.toLowerCase();
  if (suffixWord && multipliers[suffixWord]) {
    return numeric * multipliers[suffixWord];
  }

  return numeric;
};

const parsePercentageValue = (value?: string | null): number | null => {
  if (!value || isValueUnavailable(value)) {
    return null;
  }

  const normalized = value.toString().trim().toLowerCase();
  const match = normalized.match(/-?\d+(\.\d+)?/);

  if (!match) {
    return null;
  }

  let numeric = parseFloat(match[0]);

  if (Number.isNaN(numeric)) {
    return null;
  }

  if (numeric <= 1 && normalized.includes('0.')) {
    numeric *= 100;
  }

  return numeric;
};

const formatCompactCurrencyNumber = (value?: number | null) => {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return 'N/A';
  }

  const absolute = Math.abs(value);
  const suffixFormatter = (divisor: number, suffix: string) =>
    `$${(value / divisor).toFixed(1).replace(/\.0$/, '')}${suffix}`;

  if (absolute >= 1_000_000_000_000) {
    return suffixFormatter(1_000_000_000_000, 'T');
  }
  if (absolute >= 1_000_000_000) {
    return suffixFormatter(1_000_000_000, 'B');
  }
  if (absolute >= 1_000_000) {
    return suffixFormatter(1_000_000, 'M');
  }
  if (absolute >= 1_000) {
    return suffixFormatter(1_000, 'K');
  }

  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: value % 1 === 0 ? 0 : 2,
  }).format(value);
};

const truncateLabel = (label: string, limit = 48) => {
  if (label.length <= limit) {
    return label;
  }

  return `${label.slice(0, Math.max(limit - 1, 0))}…`;
};

type EstimationDatum = {
  name: string;
  fullLabel?: string;
  baseRevenue?: number;
  baseRevenueDisplay?: string;
  averageContractValue?: number;
  averageContractValueDisplay?: string;
};

const formatProbabilityDisplay = (value?: string | null): string | null => {
  const numeric = parsePercentageValue(value);

  if (numeric !== null) {
    return formatNumericPercentage(numeric);
  }

  if (!value || isValueUnavailable(value)) {
    return null;
  }

  return value;
};

const clampPercentage = (value: number) => Math.min(100, Math.max(0, value));

const formatNumericPercentage = (value: number) => {
  const clamped = clampPercentage(value);
  return `${clamped % 1 === 0 ? clamped.toFixed(0) : clamped.toFixed(1)}%`;
};

const calculateRiskScore = (probability?: string | null): number | null => {
  const numericProbability = parsePercentageValue(probability);
  if (numericProbability === null) {
    return null;
  }

  return clampPercentage(100 - numericProbability);
};

const formatRiskScoreDisplay = (probability?: string | null): string | null => {
  const riskScore = calculateRiskScore(probability);
  if (riskScore === null) {
    return null;
  }

  return formatNumericPercentage(riskScore);
};

const formatCurrencyValue = (value?: string | number | null): string => {
  if (value === undefined || value === null) {
    return 'N/A';
  }

  if (typeof value === 'string') {
    if (isValueUnavailable(value)) {
      return 'N/A';
    }
  }

  const numeric = typeof value === 'number' ? (Number.isFinite(value) ? value : null) : parseFinancialNumber(value);

  if (numeric !== null) {
    const formatter = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: Math.abs(numeric) < 1 ? 2 : 0,
    });
    return formatter.format(numeric);
  }

  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) {
      return 'N/A';
    }

    if (trimmed.startsWith('$')) {
      return trimmed;
    }

    if (/^usd\b/i.test(trimmed)) {
      return `$${trimmed.replace(/^usd\b/i, '').trim()}`;
    }

    return `$${trimmed}`;
  }

  return 'N/A';
};

const MetricCard = ({ title, value, icon, tooltip, formatAsCurrency }: MetricCardProps) => {
  const displayValue = (() => {
    if (formatAsCurrency) {
      return formatCurrencyValue(value ?? null);
    }

    if (typeof value === 'number') {
      return value.toString();
    }

    if (typeof value === 'string') {
      const trimmed = value.trim();
      return trimmed || 'N/A';
    }

    return 'N/A';
  })();

  return (
  <Card>
    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
      <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
      {tooltip ? (
        <TooltipProvider>
            <Tooltip>
                <TooltipTrigger asChild>
                    <span className="cursor-help">{icon}</span>
                </TooltipTrigger>
                <TooltipContent>
                    <p>{tooltip}</p>
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
      ) : (
        icon
      )}
    </CardHeader>
    <CardContent>
      <div className="text-2xl font-bold font-headline">{displayValue}</div>
    </CardContent>
  </Card>
  );
};

export default function Financials({ data, claims }: { data: FinancialsType, claims: ClaimsAnalysisType }) {
  const [suggestions, setSuggestions] = useState<string[] | undefined | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const projections = Array.isArray(data?.projections) ? data.projections : [];
  const claimsList = Array.isArray(claims) ? claims : [];

  const claimsChartData = claimsList
    .slice(0, 5)
    .map((claim, index): ChartDatum | null => {
      const probability = parsePercentageValue(claim.simulated_probability);
      if (probability === null) {
        return null;
      }

      return {
        name: truncateLabel(claim.claim, 40) || `Claim ${index + 1}`,
        fullLabel: claim.claim,
        value: probability,
        display: claim.simulated_probability,
      };
    })
    .filter((value): value is ChartDatum => value !== null);

  const estimationChartData = claimsList
    .slice(0, 5)
    .map((claim, index): EstimationDatum | null => {
      const assumptions = claim.simulation_assumptions ?? {};
      const baseRevenueValue = parseFinancialNumber(assumptions.base_revenue ?? null);
      const averageContractValueValue = parseFinancialNumber(assumptions.average_contract_value ?? null);

      if (baseRevenueValue === null && averageContractValueValue === null) {
        return null;
      }

      return {
        name: truncateLabel(claim.claim, 40) || `Claim ${index + 1}`,
        fullLabel: claim.claim,
        baseRevenue: baseRevenueValue ?? undefined,
        baseRevenueDisplay: formatCurrencyValue(
          baseRevenueValue ?? assumptions.base_revenue ?? null,
        ) ?? undefined,
        averageContractValue: averageContractValueValue ?? undefined,
        averageContractValueDisplay: formatCurrencyValue(
          averageContractValueValue ?? assumptions.average_contract_value ?? null,
        ) ?? undefined,
      };
    })
    .filter((value): value is EstimationDatum => value !== null);

  const handleGenerateSuggestions = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const serializedAnalysisInput = JSON.stringify(data ?? {}) + JSON.stringify(claimsList);
      const result = await generateFinancialMetricsDashboard({ analysisText: serializedAnalysisInput });
      setSuggestions(result.followUpSuggestions);
    } catch (e) {
      setError('Failed to generate suggestions. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="font-headline text-2xl mb-4">Key Metrics</h2>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <MetricCard
            title="ARR"
            value={data?.srr_mrr?.current_booked_arr}
            icon={<BarChart2 className="h-4 w-4 text-muted-foreground" />}
            formatAsCurrency
          />
          <MetricCard
            title="MRR"
            value={data?.srr_mrr?.current_mrr}
            icon={<Calendar className="h-4 w-4 text-muted-foreground" />}
            formatAsCurrency
          />
          <MetricCard
            title="Est. Burn Rate"
            value={data?.burn_and_runway?.implied_net_burn}
            icon={<HelpCircle className="h-4 w-4 text-muted-foreground" />}
            tooltip="Estimated by dividing funding ask by runway"
            formatAsCurrency
          />
          <MetricCard title="Runway" value={data?.burn_and_runway?.stated_runway} icon={<GitBranch className="h-4 w-4 text-muted-foreground" />} />
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="font-headline text-2xl flex items-center gap-3"><PiggyBank className="w-7 h-7 text-primary" />Funding</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1">
            <h4 className="font-semibold">Current Ask</h4>
            <p className="text-xl font-bold font-headline text-primary">{formatCurrencyValue(data?.burn_and_runway?.funding_ask ?? null)}</p>
          </div>
          <div className="space-y-1">
            <h4 className="font-semibold">Valuation Rationale</h4>
            <p className="text-sm text-muted-foreground">{data?.valuation_rationale || 'N/A'}</p>
          </div>
          <div className="space-y-1">
            <h4 className="font-semibold">Previous Funding</h4>
            <p className="text-sm text-muted-foreground">{data?.funding_history || 'N/A'}</p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="font-headline text-2xl flex items-center gap-3"><Briefcase className="w-7 h-7 text-primary" />Recurring Revenue Summary</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3">
            <div className="flex items-center justify-between rounded-md bg-secondary/50 p-3">
              <span className="text-sm font-medium text-muted-foreground">ARR</span>
              <span className="text-lg font-semibold font-headline">{formatCurrencyValue(data?.srr_mrr?.current_booked_arr ?? null)}</span>
            </div>
            <div className="flex items-center justify-between rounded-md bg-secondary/50 p-3">
              <span className="text-sm font-medium text-muted-foreground">MRR</span>
              <span className="text-lg font-semibold font-headline">{formatCurrencyValue(data?.srr_mrr?.current_mrr ?? null)}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="font-headline text-2xl flex items-center gap-3"><Calendar className="w-7 h-7 text-primary" />Financial Projections</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {projections.length > 0 ? (
            <ul className="space-y-2">
              {projections.map((projection) => (
                <li key={projection.year} className="flex justify-between items-center p-2 rounded-md bg-secondary/30">
                  <span className="font-medium">{projection.year}</span>
                  <span className="font-bold text-lg font-headline text-primary">{formatCurrencyValue(projection.revenue)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground">No financial projections were provided in the memo.</p>
          )}
        </CardContent>
      </Card>

      {claimsChartData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="font-headline text-2xl flex items-center gap-3"><BarChart2 className="w-7 h-7 text-primary" />Claim Confidence Snapshot</CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={claimsChartData} layout="vertical" margin={{ top: 8, bottom: 8, left: 16, right: 16 }}>
                <CartesianGrid horizontal strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  type="number"
                  domain={[0, 100]}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                  tickFormatter={(value) => formatNumericPercentage(Number(value))}
                />
                <YAxis
                  dataKey="name"
                  type="category"
                  width={200}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                />
                <RechartsTooltip
                  formatter={(_, __, payload) => {
                    const formatted = payload?.payload as ChartDatum | undefined;
                    const numericValue = formatted?.value ?? null;
                    const fallback = numericValue === null ? 'N/A' : formatNumericPercentage(numericValue);
                    return [formatted?.display ?? fallback, 'Simulated Probability'];
                  }}
                  labelFormatter={(_, payload) => (payload?.[0]?.payload as ChartDatum | undefined)?.fullLabel ?? ''}
                  contentStyle={{
                    backgroundColor: 'hsl(var(--background))',
                    borderRadius: '0.75rem',
                    border: '1px solid hsl(var(--border))',
                    maxWidth: '22rem',
                  }}
                />
                <Bar dataKey="value" fill="hsl(var(--accent))" radius={[0, 12, 12, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {estimationChartData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="font-headline text-2xl flex items-center gap-3"><Brain className="w-7 h-7 text-primary" />Model Estimation Snapshot</CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={estimationChartData} layout="vertical" margin={{ top: 12, bottom: 12, left: 16, right: 16 }}>
                <CartesianGrid horizontal strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  type="number"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                  tickFormatter={(value) => formatCompactCurrencyNumber(Number(value))}
                />
                <YAxis
                  dataKey="name"
                  type="category"
                  width={200}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                />
                <RechartsTooltip
                  formatter={(value: number, name, payload) => {
                    const formatted = payload?.payload as EstimationDatum | undefined;
                    if (name === 'baseRevenue') {
                      return [
                        formatted?.baseRevenueDisplay ?? formatCompactCurrencyNumber(value),
                        'Base Revenue',
                      ];
                    }
                    return [
                      formatted?.averageContractValueDisplay ?? formatCompactCurrencyNumber(value),
                      'Average Contract Value',
                    ];
                  }}
                  labelFormatter={(_, payload) => (payload?.[0]?.payload as EstimationDatum | undefined)?.fullLabel ?? ''}
                  contentStyle={{
                    backgroundColor: 'hsl(var(--background))',
                    borderRadius: '0.75rem',
                    border: '1px solid hsl(var(--border))',
                    maxWidth: '22rem',
                  }}
                />
                <Legend
                  verticalAlign="top"
                  wrapperStyle={{ paddingBottom: '0.5rem', fontSize: '0.75rem' }}
                />
                <Bar dataKey="baseRevenue" fill="hsl(var(--primary))" radius={[0, 12, 12, 0]} name="Base Revenue" />
                <Bar dataKey="averageContractValue" fill="hsl(var(--accent))" radius={[0, 12, 12, 0]} name="Average Contract Value" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {claimsList.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {claimsList.map((claim, index) => {
            const probabilityDisplay = formatProbabilityDisplay(claim.simulated_probability);
            const riskScoreDisplay = formatRiskScoreDisplay(claim.simulated_probability);
            return (
              <Card key={index}>
                <CardHeader>
                  <CardTitle className="font-headline text-xl flex items-center gap-3"><Target className="w-6 h-6 text-primary" />Claim Analysis: {claim.claim}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between items-center bg-secondary/50 p-4 rounded-lg">
                    <span className="font-semibold text-lg">Simulated Probability</span>
                    <span className="text-3xl font-bold font-headline text-accent">
                      {probabilityDisplay ?? 'N/A'}
                    </span>
                  </div>
                  {probabilityDisplay === null && (
                    <p className="text-xs text-muted-foreground">
                      The model did not return a probability estimate for this claim.
                    </p>
                  )}
                  <div className="flex justify-between items-center bg-secondary/30 p-4 rounded-lg">
                    <span className="font-semibold text-lg">Risk Score</span>
                    <span className="text-2xl font-bold font-headline text-destructive">
                      {riskScoreDisplay ?? 'N/A'}
                    </span>
                  </div>
                  {riskScoreDisplay !== null && (
                    <p className="text-xs text-muted-foreground">
                      Higher percentages represent elevated diligence risk for this claim.
                    </p>
                  )}
                  <div className="space-y-1">
                    <h4 className="font-semibold">Result</h4>
                    <p className="text-sm text-muted-foreground">{claim.result}</p>
                  </div>
                  <p className="text-sm text-muted-foreground"><span className="font-semibold">Analysis Method:</span> {claim.analysis_method}</p>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="font-headline text-2xl flex items-center gap-3"><Sparkles className="w-7 h-7 text-accent"/>AI-Powered Investigation Suggestions</CardTitle>
          <CardDescription>Generate AI suggestions for follow-up questions based on financial projections.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!suggestions && !isLoading && (
            <Button onClick={handleGenerateSuggestions} disabled={isLoading}>
               {isLoading ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Briefcase className="mr-2 h-4 w-4" />
              )}
              Generate Suggestions
            </Button>
          )}
          {isLoading && (
            <div className="flex items-center space-x-2 text-muted-foreground">
              <Loader2 className="h-5 w-5 animate-spin" />
              <span>Thinking of smart questions...</span>
            </div>
          )}
          {error && <Alert variant="destructive"><AlertTitle>Error</AlertTitle><AlertDescription>{error}</AlertDescription></Alert>}
          {suggestions && (
            <Alert>
              <Sparkles className="h-4 w-4" />
              <AlertTitle className="font-headline">Follow-up Investigations</AlertTitle>
              <AlertDescription>
                <ul className="list-disc pl-5 mt-2 space-y-1">
                  {suggestions.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
