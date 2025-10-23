
import type { CompanyOverview as CompanyOverviewType } from '@/lib/types';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Briefcase, Building, Cpu, Users, GraduationCap, Building2, User } from 'lucide-react';

type CompanyOverviewProps = {
  data?: CompanyOverviewType | null;
};

const FALLBACK_TEXT = 'Not available';

const normalizeText = (value?: string | null): string => {
  if (typeof value !== 'string') {
    return FALLBACK_TEXT;
  }

  const trimmed = value.trim();
  return trimmed || FALLBACK_TEXT;
};

const toSafeFounders = (value: CompanyOverviewType['founders'] | undefined): CompanyOverviewType['founders'] => {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .filter((founder) => Boolean(founder))
    .map((founder, index) => ({
      name: normalizeText(founder?.name ?? `Founder ${index + 1}`),
      education: founder?.education ?? null,
      previous_ventures: founder?.previous_ventures ?? null,
      professional_background: founder?.professional_background ?? null,
      email: founder?.email ?? null,
    }));
};

export default function CompanyOverview({ data }: CompanyOverviewProps) {
  const safeName = normalizeText(data?.name);
  const safeSector = normalizeText(data?.sector);
  const safeTechnology = normalizeText(data?.technology);
  const founders = toSafeFounders(data?.founders);

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle className="font-headline text-3xl flex items-center gap-3">
            <Building className="w-8 h-8 text-primary" />
            {safeName}
          </CardTitle>
          <CardDescription className="text-lg">{safeSector}</CardDescription>
        </CardHeader>
        <CardContent>
          <h3 className="font-headline text-xl mb-4 flex items-center gap-2"><Cpu className="w-5 h-5 text-muted-foreground"/>Technology</h3>
          <p className="text-muted-foreground">{safeTechnology}</p>
        </CardContent>
      </Card>

      <div>
        <h2 className="font-headline text-2xl mb-4 flex items-center gap-3"><Users className="w-7 h-7 text-primary"/>Founders</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {founders.length > 0 ? (
            founders.map((founder, index) => (
              <Card key={`${founder.name}-${index}`} className="flex flex-col">
                <CardHeader className="flex flex-row items-center gap-4">
                  <Avatar className="h-16 w-16">
                    <AvatarFallback className="text-xl bg-secondary">
                      <User className="w-8 h-8 text-muted-foreground" />
                    </AvatarFallback>
                  </Avatar>
                  <div>
                    <CardTitle className="font-headline text-xl">{founder.name}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4 flex-1">
                  {founder.education && (
                    <div className="flex items-start gap-3">
                      <GraduationCap className="w-5 h-5 mt-1 text-muted-foreground flex-shrink-0" />
                      <div>
                        <h4 className="font-semibold">Education</h4>
                        <p className="text-muted-foreground text-sm">{founder.education}</p>
                      </div>
                    </div>
                  )}
                  {founder.professional_background && (
                    <div className="flex items-start gap-3">
                      <Briefcase className="w-5 h-5 mt-1 text-muted-foreground flex-shrink-0" />
                      <div>
                        <h4 className="font-semibold">Professional Background</h4>
                        <p className="text-muted-foreground text-sm">{founder.professional_background}</p>
                      </div>
                    </div>
                  )}
                  {founder.previous_ventures && (
                    <div className="flex items-start gap-3">
                      <Building2 className="w-5 h-5 mt-1 text-muted-foreground flex-shrink-0" />
                      <div>
                        <h4 className="font-semibold">Previous Ventures</h4>
                        <p className="text-muted-foreground text-sm">{founder.previous_ventures}</p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))
          ) : (
            <Card className="col-span-full">
              <CardHeader>
                <CardTitle className="font-headline text-xl">Founder details</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">{FALLBACK_TEXT}</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
