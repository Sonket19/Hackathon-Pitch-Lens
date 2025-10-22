import {z} from 'zod';
import type {AnalysisData, MemoV1} from '@/lib/types';

const ChatMessageSchema = z.object({
  role: z.enum(['user', 'model']),
  content: z.string(),
});

const StartupInterviewerInputSchema = z.object({
  analysisData: z.string().describe('The JSON string of the full startup analysis data.'),
  history: z.array(ChatMessageSchema).describe('The conversation history.'),
});
export type StartupInterviewerInput = z.infer<typeof StartupInterviewerInputSchema>;

const StartupInterviewerOutputSchema = z.object({
  message: z.string().describe("The chatbot's response."),
});
export type StartupInterviewerOutput = z.infer<typeof StartupInterviewerOutputSchema>;

const TOPIC_SEQUENCE = ['mission', 'market', 'product', 'goToMarket', 'financials', 'risk'] as const;

type TopicId = typeof TOPIC_SEQUENCE[number];

type SafeMemo = Partial<MemoV1> & {risk_metrics?: Partial<MemoV1['risk_metrics']>};

type ParsedAnalysis = {
  companyName?: string;
  sector?: string;
  memo?: SafeMemo;
};

function safeParseAnalysis(json: string): ParsedAnalysis {
  try {
    const parsed = JSON.parse(json) as Partial<AnalysisData>;
    return {
      companyName: parsed?.metadata?.company_name ?? parsed?.memo?.draft_v1?.company_overview?.name,
      sector: parsed?.metadata?.sector ?? parsed?.memo?.draft_v1?.company_overview?.sector,
      memo: parsed?.memo?.draft_v1 ?? (parsed?.memo as SafeMemo),
    };
  } catch (error) {
    console.warn('Failed to parse analysis JSON for chatbot', error);
    return {};
  }
}

function extractAskedTopics(history: StartupInterviewerInput['history']): Set<TopicId> {
  const asked = new Set<TopicId>();
  history
    .filter(message => message.role === 'model')
    .forEach(message => {
      for (const topic of TOPIC_SEQUENCE) {
        if (message.content.toLowerCase().includes(topic)) {
          asked.add(topic);
        }
      }
    });
  return asked;
}

function selectNextTopic(askedTopics: Set<TopicId>): TopicId {
  for (const topic of TOPIC_SEQUENCE) {
    if (!askedTopics.has(topic)) {
      return topic;
    }
  }
  return 'risk';
}

function formatCompanyIntro(parsed: ParsedAnalysis): string {
  const parts: string[] = [];
  if (parsed.companyName) {
    parts.push(parsed.companyName);
  } else {
    parts.push('this startup');
  }
  if (parsed.sector) {
    parts.push(`in the ${parsed.sector} space`);
  }
  return parts.join(' ');
}

function buildMissionQuestion(parsed: ParsedAnalysis): string {
  const intro = formatCompanyIntro(parsed);
  const differentiator = parsed.memo?.business_model?.scalability ?? parsed.memo?.business_model?.revenue_streams;
  const focus = differentiator ? `I noticed your deck highlights ${differentiator.toLowerCase()}.` : 'I would love to understand your focus a bit more.';
  return `Hi there! Thanks for sharing ${intro}. ${focus} What near-term milestone are you targeting over the next two quarters?`;
}

function buildMarketQuestion(parsed: ParsedAnalysis): string {
  const market = parsed.memo?.market_analysis?.industry_size_and_growth?.total_addressable_market?.value;
  const commentary = parsed.memo?.market_analysis?.industry_size_and_growth?.commentary;
  const detail = market ?? commentary;
  const context = detail ? `You mentioned a market opportunity around ${detail}.` : 'The materials outline a sizeable market opportunity.';
  return `${context} Which customer segment is proving the easiest to win today, and why?`;
}

function buildProductQuestion(parsed: ParsedAnalysis): string {
  const differentiation = parsed.memo?.business_model?.scalability ?? parsed.memo?.business_model?.unit_economics?.customer_lifetime_value_ltv;
  const hook = differentiation ? `Given your advantage around ${differentiation.toLowerCase()},` : 'Thinking about your product moat,';
  return `${hook} what is the hardest capability for competitors to replicate right now?`;
}

function buildGtmQuestion(parsed: ParsedAnalysis): string {
  const recentNews = parsed.memo?.market_analysis?.recent_news;
  const reference = recentNews ? `I saw mention of ${recentNews}.` : 'You referenced a growing pipeline.';
  return `${reference} Can you walk me through the most reliable channel for new deals and how you scale it efficiently?`;
}

function buildFinancialQuestion(parsed: ParsedAnalysis): string {
  const burn = parsed.memo?.financials?.burn_and_runway?.implied_net_burn;
  const runway = parsed.memo?.financials?.burn_and_runway?.stated_runway;
  const tag = [burn, runway].filter(Boolean).join(' and ');
  const framing = tag ? `With ${tag} in the plan,` : 'Looking at your current plan,';
  return `${framing} what are the key levers that could extend runway without slowing growth?`;
}

function buildRiskQuestion(parsed: ParsedAnalysis): string {
  const riskScore = parsed.memo?.risk_metrics?.composite_risk_score;
  const interpretation = parsed.memo?.risk_metrics?.score_interpretation;
  const baseline = typeof riskScore === 'number' ? `Your risk score sits around ${riskScore}.` : 'You shared an overall risk summary.';
  const followUp = interpretation ? `You described it as "${interpretation}".` : '';
  return `${baseline} ${followUp} What keeps you up at night about executing this plan?`;
}

const QUESTION_BUILDERS: Record<TopicId, (parsed: ParsedAnalysis) => string> = {
  mission: buildMissionQuestion,
  market: buildMarketQuestion,
  product: buildProductQuestion,
  goToMarket: buildGtmQuestion,
  financials: buildFinancialQuestion,
  risk: buildRiskQuestion,
};

export async function interviewStartup(input: StartupInterviewerInput): Promise<StartupInterviewerOutput> {
  const parsedInput = StartupInterviewerInputSchema.parse(input);
  const parsedAnalysis = safeParseAnalysis(parsedInput.analysisData);
  const askedTopics = extractAskedTopics(parsedInput.history);
  const nextTopic = selectNextTopic(askedTopics);
  const builder = QUESTION_BUILDERS[nextTopic] ?? buildMissionQuestion;
  const message = builder(parsedAnalysis);
  return StartupInterviewerOutputSchema.parse({message});
}
