"use client";

import { useMemo } from 'react';
import { Button } from '@/components/ui/button';
import type { AnalysisData } from '@/lib/types';

const EMAIL_REGEX = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi;

const collectEmails = (value: unknown, push: (email: string) => void): void => {
  if (!value) {
    return;
  }

  if (typeof value === 'string') {
    const matches = value.match(EMAIL_REGEX);
    if (matches) {
      matches.forEach(match => {
        const trimmed = match.trim();
        if (trimmed) {
          push(trimmed);
        }
      });
    }
    return;
  }

  if (Array.isArray(value)) {
    value.forEach(item => collectEmails(item, push));
    return;
  }

  if (typeof value === 'object') {
    Object.values(value as Record<string, unknown>).forEach(item => collectEmails(item, push));
  }
};

type Props = {
  analysisData: AnalysisData;
  className?: string;
};

const coerce = (value: unknown): string => (typeof value === 'string' ? value.trim() : '');

const sanitizeProductName = (value: string): string | undefined => {
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }
  const normalized = trimmed.replace(/\s+/g, ' ');
  const wordCount = normalized.split(' ').length;
  if (normalized.length > 60 || wordCount > 6) {
    return undefined;
  }
  return normalized;
};

const PLACEHOLDER_STRINGS = new Set(['', 'n/a', 'na', 'none', 'not available', 'not specified', 'unknown']);

const isPlaceholder = (value: unknown): boolean => {
  if (value === null || value === undefined) {
    return true;
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    return normalized.length === 0 || PLACEHOLDER_STRINGS.has(normalized);
  }
  if (Array.isArray(value)) {
    return value.length === 0 || value.every(item => isPlaceholder(item));
  }
  if (typeof value === 'object') {
    return Object.keys(value as Record<string, unknown>).length === 0;
  }
  return false;
};

type OutreachContext = {
  founderEmails: string[];
  companyName?: string;
  productName?: string;
  displayName: string;
  fallbackCompany: string;
};

const useFounderOutreachContext = (analysisData: AnalysisData): OutreachContext => {
  return useMemo(() => {
    const rawPublic = analysisData.public_data as unknown;
    let publicData: Record<string, unknown> = {};
    if (rawPublic) {
      if (typeof rawPublic === 'string') {
        try {
          const parsed = JSON.parse(rawPublic);
          if (typeof parsed === 'object' && parsed !== null) {
            publicData = parsed as Record<string, unknown>;
          }
        } catch (error) {
          console.error('Failed to parse public_data payload for email CTA', error);
        }
      } else if (typeof rawPublic === 'object') {
        publicData = rawPublic as Record<string, unknown>;
      }
    }

    const namesMeta = (analysisData.metadata?.names ?? {}) as Partial<{
      company: string;
      product: string;
      display: string;
    }>;

    const companyName =
      coerce(namesMeta.company) ||
      coerce(analysisData.metadata?.company_legal_name) ||
      coerce(analysisData.metadata?.company_name) ||
      coerce(analysisData.memo?.draft_v1?.company_overview?.name) ||
      undefined;

    const rawProductName =
      coerce(namesMeta.product) ||
      coerce((analysisData.metadata as { product_name?: unknown } | undefined)?.product_name) ||
      '';

    const memoProductCandidate = coerce(analysisData.memo?.draft_v1?.company_overview?.technology);

    const productName =
      sanitizeProductName(rawProductName) ||
      (memoProductCandidate ? sanitizeProductName(memoProductCandidate) : undefined);

    const displayName =
      coerce(namesMeta.display) ||
      coerce(analysisData.metadata?.display_name) ||
      companyName ||
      productName ||
      'the company';

    const founderEmails: string[] = [];
    const seen = new Set<string>();
    const push = (email: string) => {
      const trimmed = email.trim();
      if (!trimmed) {
        return;
      }
      const key = trimmed.toLowerCase();
      if (seen.has(key)) {
        return;
      }
      seen.add(key);
      founderEmails.push(trimmed);
    };

    const metadata = analysisData.metadata as {
      founder_emails?: unknown;
      contact_email?: unknown;
      founder_contacts?: unknown;
    } | undefined;

    if (metadata) {
      collectEmails(metadata.founder_emails, push);
      collectEmails(metadata.contact_email, push);
      collectEmails(metadata.founder_contacts, push);
    }

    collectEmails(analysisData.memo?.draft_v1?.company_overview?.founders, push);
    collectEmails(analysisData.memo?.draft_v1, push);
    collectEmails(publicData, push);

    const fallbackCompany = companyName || displayName || productName || 'your company';

    return {
      founderEmails,
      companyName,
      productName,
      displayName,
      fallbackCompany,
    };
  }, [analysisData]);
};

const normaliseQuestion = (question: string): string => {
  const trimmed = question.trim();
  if (!trimmed) {
    return '';
  }
  if (/[.?!)]$/.test(trimmed)) {
    return trimmed;
  }
  return `${trimmed}?`;
};

const extractQuestionArray = (value: unknown): string[] => {
  if (!value) {
    return [];
  }

  if (Array.isArray(value)) {
    return value
      .map(item => (typeof item === 'string' ? normaliseQuestion(item) : ''))
      .filter(Boolean);
  }

  if (typeof value === 'string') {
    if (!value.trim()) {
      return [];
    }
    const parts = value
      .split(/\r?\n|[•\-]\s+/)
      .map(part => normaliseQuestion(part))
      .filter(Boolean);
    return parts.length > 0 ? parts : [normaliseQuestion(value)];
  }

  return [];
};

const deriveRemainingCoverageQuestions = (
  analysisData: AnalysisData,
  context: OutreachContext,
): string[] => {
  const questions = new Set<string>();
  const push = (value: unknown) => {
    extractQuestionArray(value).forEach(question => {
      if (question) {
        questions.add(question);
      }
    });
  };

  const metadata = analysisData.metadata as Record<string, unknown> | undefined;
  if (metadata) {
    push(metadata['coverage_questions']);
    push(metadata['remaining_coverage_questions']);
    push(metadata['follow_up_questions']);
    push(metadata['data_requests']);
  }

  const memo = analysisData.memo?.draft_v1;
  const memoAny = memo as Record<string, unknown> | undefined;
  const financialsAny = memoAny?.financials as Record<string, unknown> | undefined;

  if (financialsAny) {
    push(financialsAny['followUpSuggestions']);
    push(financialsAny['follow_up_suggestions']);
    push(financialsAny['follow_up_questions']);
    push(financialsAny['ai_followups']);
    push(financialsAny['outstanding_questions']);
  }

  push(memoAny?.['outstanding_questions']);
  push(memoAny?.['remaining_questions']);

  if (questions.size > 0) {
    return Array.from(questions);
  }

  if (!memo) {
    return [];
  }

  const fallbackName = context.displayName || context.companyName || 'your company';
  const financials = memo.financials;

  const addQuestion = (value: unknown, prompt: string) => {
    if (isPlaceholder(value)) {
      questions.add(prompt);
    }
  };

  if (financials) {
    addQuestion(
      financials.funding_history,
      `Could you share a detailed funding history for ${fallbackName}?`,
    );
    addQuestion(
      financials.valuation_rationale,
      `What is the valuation rationale or supporting metrics for ${fallbackName}'s current raise?`,
    );
    const burn = financials.burn_and_runway;
    if (burn) {
      addQuestion(
        burn.funding_ask,
        `What is the precise funding ask in USD for ${fallbackName}'s current round?`,
      );
      addQuestion(
        burn.stated_runway,
        `How many months of runway will this raise provide for ${fallbackName}?`,
      );
      addQuestion(
        burn.implied_net_burn,
        `What is the company's current monthly net burn rate in USD?`,
      );
    }
    const recurring = financials.srr_mrr;
    if (recurring) {
      addQuestion(
        recurring.current_booked_arr,
        `Can you confirm the latest booked ARR figures for ${fallbackName}?`,
      );
      addQuestion(
        recurring.current_mrr,
        `Can you share the current MRR for ${fallbackName}?`,
      );
    }
    const projections = financials.projections;
    if (Array.isArray(projections) && projections.length > 0) {
      projections.forEach((projection, index) => {
        if (!projection) {
          return;
        }
        const label = projection.year && !isPlaceholder(projection.year)
          ? projection.year
          : `projection ${index + 1}`;
        if (isPlaceholder(projection.revenue)) {
          questions.add(`What is the projected revenue for ${label}?`);
        }
      });
    }
  }

  const overview = memo.company_overview;
  if (overview) {
    addQuestion(
      overview.technology,
      `Could you provide a succinct explanation of the core technology behind ${fallbackName}?`,
    );
    addQuestion(
      overview.sector,
      `Which sector does ${fallbackName} primarily operate within?`,
    );
    if (Array.isArray(overview.founders)) {
      overview.founders.forEach(founder => {
        if (!founder) {
          return;
        }
        const name = typeof founder === 'object' && founder && 'name' in founder ? (founder as { name?: string }).name : undefined;
        const display = name && name.trim() ? name.trim() : 'one of the founding team members';
        if (typeof founder === 'object' && founder !== null) {
          const founderRecord = founder as Record<string, unknown>;
          addQuestion(
            founderRecord['professional_background'],
            `Could you share ${display}'s professional background?`,
          );
          addQuestion(
            founderRecord['education'],
            `What is ${display}'s educational background?`,
          );
          addQuestion(
            founderRecord['previous_ventures'],
            `Has ${display} founded or worked on prior ventures you can detail?`,
          );
        }
      });
    }
  }

  const market = memo.market_analysis;
  if (market) {
    const industry = market.industry_size_and_growth;
    if (industry) {
      addQuestion(
        industry.total_addressable_market?.value,
        `Can you quantify the total addressable market (TAM) that ${fallbackName} is targeting?`,
      );
      addQuestion(
        industry.serviceable_obtainable_market?.value,
        `What is the serviceable obtainable market (SOM) for ${fallbackName}?`,
      );
      addQuestion(
        industry.total_addressable_market?.cagr,
        `What CAGR assumptions are you using for the target market?`,
      );
    }
    addQuestion(
      market.recent_news,
      `Are there any notable recent news items about ${fallbackName} or its market we should review?`,
    );
    if (Array.isArray(market.sub_segment_opportunities)) {
      const hasMeaningfulOpportunities = market.sub_segment_opportunities.some(
        opportunity => !isPlaceholder(opportunity),
      );
      if (!hasMeaningfulOpportunities) {
        questions.add(`Could you outline any sub-segment opportunities ${fallbackName} is pursuing?`);
      }
    }
  }

  const businessModel = memo.business_model;
  if (businessModel) {
    addQuestion(
      businessModel.revenue_streams,
      `Can you describe the primary revenue streams for ${fallbackName}?`,
    );
    addQuestion(
      businessModel.pricing,
      `What pricing model does ${fallbackName} use for its offering?`,
    );
    addQuestion(
      businessModel.scalability,
      `How does ${fallbackName} plan to scale operations over the next 12-18 months?`,
    );
    const unitEconomics = businessModel.unit_economics;
    if (unitEconomics) {
      addQuestion(
        unitEconomics.customer_lifetime_value_ltv,
        `What is the current LTV estimate for ${fallbackName}'s customers?`,
      );
      addQuestion(
        unitEconomics.customer_acquisition_cost_cac,
        `What is the latest CAC for ${fallbackName}?`,
      );
    }
  }

  const risk = memo.risk_metrics;
  if (risk) {
    addQuestion(
      risk.narrative_justification,
      `Could you provide more context supporting the current diligence risk assessment for ${fallbackName}?`,
    );
  }

  return Array.from(questions);
};

export function ConnectFoundersButton({ analysisData, className }: Props) {
  const context = useFounderOutreachContext(analysisData);

  const gmailLink = useMemo(() => {
    const hasSpecificProduct = Boolean(context.productName);
    const mailSubject = hasSpecificProduct
      ? `Pitch Discussion - ${context.productName}`
      : 'Pitch Discussion';

    const mailBodyLines = hasSpecificProduct
      ? [
          'Hi,',
          '',
          `I'd love to schedule time to discuss about (${context.productName}) from your company (${context.fallbackCompany}).`,
          'Are you available for a 30-minute call this week to cover product traction, go-to-market, and financial plans?',
          '',
          'Looking forward to the conversation.',
          '',
          'Best regards,',
          '[Your Name]',
        ]
      : [
          'Hi,',
          '',
          "I'd love to schedule time to discuss further about your pitch.",
          'Are you available for a 30-minute call this week to cover product traction, go-to-market, and financial plans?',
          '',
          'Looking forward to the conversation.',
          '',
          'Best regards,',
          '[Your Name]',
        ];

    const gmailParams = new URLSearchParams({
      view: 'cm',
      fs: '1',
      su: mailSubject,
      body: mailBodyLines.join('\n'),
    });

    if (context.founderEmails.length > 0) {
      gmailParams.set('to', context.founderEmails.join(','));
    }

    return `https://mail.google.com/mail/?${gmailParams.toString()}`;
  }, [context.fallbackCompany, context.founderEmails, context.productName]);

  return (
    <Button asChild variant="secondary" className={className}>
      <a href={gmailLink} target="_blank" rel="noopener noreferrer">
        Connect with founders
      </a>
    </Button>
  );
}

export function RemainingCoverageButton({ analysisData, className }: Props) {
  const context = useFounderOutreachContext(analysisData);

  const { gmailLink, hasQuestions } = useMemo(() => {
    const questions = deriveRemainingCoverageQuestions(analysisData, context);

    if (questions.length === 0) {
      return { gmailLink: null, hasQuestions: false };
    }

    const numbered = questions.map((question, index) => `${index + 1}. ${question}`);
    const mailBodyLines = [
      'Hi,',
      '',
      `As we wrap up our review of ${context.displayName}, we surfaced a few remaining coverage questions:`,
      '',
      ...numbered,
      '',
      'Any detail or supporting metrics you can share would be appreciated.',
      '',
      'Best regards,',
      '[Your Name]',
    ];

    const gmailParams = new URLSearchParams({
      view: 'cm',
      fs: '1',
      su: `Remaining Coverage Questions - ${context.displayName}`,
      body: mailBodyLines.join('\n'),
    });

    if (context.founderEmails.length > 0) {
      gmailParams.set('to', context.founderEmails.join(','));
    }

    return {
      gmailLink: `https://mail.google.com/mail/?${gmailParams.toString()}`,
      hasQuestions: true,
    };
  }, [analysisData, context]);

  if (!hasQuestions || !gmailLink) {
    return (
      <Button variant="outline" className={className} disabled>
        Remaining Coverage
      </Button>
    );
  }

  return (
    <Button asChild variant="outline" className={className}>
      <a href={gmailLink} target="_blank" rel="noopener noreferrer">
        Remaining Coverage
      </a>
    </Button>
  );
}
