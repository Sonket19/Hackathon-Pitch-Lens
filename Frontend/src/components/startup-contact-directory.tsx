"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import type { AnalysisData, Founder } from "@/lib/types";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Mail,
  Phone,
  Calendar as CalendarIcon,
  ArrowLeft,
  UserRound,
} from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { format } from "date-fns";
import { cn } from "@/lib/utils";

const EMAIL_REGEX = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi;

type FounderContact = {
  id: string;
  name: string;
  companyName: string;
  emails: string[];
  phones: string[];
};

type StartupContactDirectoryProps = {
  analysisData: AnalysisData | null;
  startupId: string;
  initialError?: string | null;
};

const collectEmails = (value: unknown, push: (email: string) => void): void => {
  if (!value) return;

  if (typeof value === "string") {
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

  if (typeof value === "object") {
    Object.values(value as Record<string, unknown>).forEach(item => collectEmails(item, push));
  }
};

const collectPhoneNumbers = (
  value: unknown,
  push: (phone: string) => void,
  hint = "",
): void => {
  if (!value) return;

  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed || trimmed.includes("@")) {
      return;
    }
    const digits = trimmed.replace(/\D/g, "");
    if (digits.length >= 7 || hint.toLowerCase().includes("phone")) {
      push(trimmed);
    }
    return;
  }

  if (Array.isArray(value)) {
    value.forEach(item => collectPhoneNumbers(item, push, hint));
    return;
  }

  if (typeof value === "object") {
    Object.entries(value as Record<string, unknown>).forEach(([key, val]) => {
      const nextHint = [hint, key].filter(Boolean).join(".");
      collectPhoneNumbers(val, push, nextHint);
    });
  }
};

const normaliseString = (value: unknown): string => (typeof value === "string" ? value.trim() : "");

const deriveNameFromEmail = (email: string): string | undefined => {
  const [localPart] = email.split("@");
  if (!localPart) return undefined;
  const words = localPart
    .replace(/\.+/g, " ")
    .replace(/_/g, " ")
    .split(" ")
    .map(part => part.trim())
    .filter(Boolean)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1));
  if (words.length === 0) return undefined;
  return words.join(" ");
};

const ensureParsedObject = (value: unknown): Record<string, unknown> => {
  if (!value) return {};
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === "object") {
        return parsed as Record<string, unknown>;
      }
    } catch (error) {
      console.error("Failed to parse JSON payload while building contact directory", error);
      return {};
    }
  }
  if (typeof value === "object") {
    return value as Record<string, unknown>;
  }
  return {};
};

const buildFounderDirectory = (analysisData: AnalysisData): FounderContact[] => {
  const companyName =
    normaliseString(analysisData.metadata?.display_name) ||
    normaliseString(analysisData.metadata?.company_name) ||
    normaliseString(analysisData.metadata?.company_legal_name) ||
    "Unknown Company";

  const directory = new Map<string, { name: string; emails: Set<string>; phones: Set<string> }>();
  let fallbackIndex = 1;

  const ensureRecord = (rawName?: string): { name: string; emails: Set<string>; phones: Set<string> } => {
    let name = normaliseString(rawName);
    if (!name) {
      name = `Founder ${fallbackIndex++}`;
    }
    const key = name.toLowerCase();
    if (!directory.has(key)) {
      directory.set(key, { name, emails: new Set<string>(), phones: new Set<string>() });
    }
    return directory.get(key)!;
  };

  const addEmailToRecord = (nameCandidate: string | undefined, email: string) => {
    const trimmedEmail = email.trim();
    if (!trimmedEmail) return;
    const record = ensureRecord(nameCandidate || deriveNameFromEmail(trimmedEmail));
    record.emails.add(trimmedEmail);
  };

  const addPhoneToRecord = (nameCandidate: string | undefined, phone: string) => {
    const trimmedPhone = phone.trim();
    if (!trimmedPhone) return;
    const record = ensureRecord(nameCandidate);
    record.phones.add(trimmedPhone);
  };

  const memoFounders = analysisData.memo?.draft_v1?.company_overview?.founders as Founder[] | undefined;
  memoFounders?.forEach((founder, index) => {
    const name = normaliseString(founder.name) || `Founder ${index + 1}`;
    const record = ensureRecord(name);
    const email = normaliseString(founder.email);
    if (email) {
      record.emails.add(email);
    }
  });

  const founderNames = Array.isArray(analysisData.metadata?.founder_names)
    ? (analysisData.metadata?.founder_names as unknown[])
        .map(normaliseString)
        .filter(Boolean)
    : [];
  founderNames.forEach(name => {
    ensureRecord(name);
  });

  const founderEmails = Array.isArray(analysisData.metadata?.founder_emails)
    ? (analysisData.metadata?.founder_emails as unknown[])
        .map(normaliseString)
        .filter(Boolean)
    : [];

  if (founderEmails.length === founderNames.length && founderEmails.length > 0) {
    founderEmails.forEach((email, idx) => addEmailToRecord(founderNames[idx], email));
  } else {
    founderEmails.forEach(email => addEmailToRecord(undefined, email));
  }

  const contactEmail = normaliseString(analysisData.metadata?.contact_email);
  if (contactEmail) {
    addEmailToRecord(founderNames[0], contactEmail);
  }

  const metadataContacts = ensureParsedObject(analysisData.metadata?.founder_contacts);
  const publicPayload = ensureParsedObject(analysisData.public_data);
  const publicContacts = ensureParsedObject(publicPayload.founder_contacts);

  const metadataEmails = new Set<string>();
  collectEmails(metadataContacts, email => metadataEmails.add(email));
  metadataEmails.forEach(email => addEmailToRecord(undefined, email));

  const publicEmails = new Set<string>();
  collectEmails(publicContacts, email => publicEmails.add(email));
  publicEmails.forEach(email => addEmailToRecord(undefined, email));

  const phoneNumbers = new Map<string, { displayName?: string; phones: Set<string> }>();
  const registerPhone = (nameCandidate: string | undefined, phone: string) => {
    const trimmed = phone.trim();
    if (!trimmed) return;
    const key = nameCandidate ? nameCandidate.toLowerCase() : "__primary__";
    if (!phoneNumbers.has(key)) {
      phoneNumbers.set(key, { displayName: nameCandidate, phones: new Set() });
    }
    const entry = phoneNumbers.get(key)!;
    if (nameCandidate && !entry.displayName) {
      entry.displayName = nameCandidate;
    }
    entry.phones.add(trimmed);
  };

  collectPhoneNumbers(metadataContacts, phone => registerPhone(undefined, phone));
  collectPhoneNumbers(publicContacts, phone => registerPhone(undefined, phone));

  const metadataPhoneCandidates = [
    normaliseString((analysisData.metadata as { contact_phone?: unknown } | undefined)?.contact_phone),
    normaliseString((analysisData.metadata as { contact_number?: unknown } | undefined)?.contact_number),
  ].filter(Boolean);
  metadataPhoneCandidates.forEach(phone => registerPhone(founderNames[0], phone));

  const memoFounderPhones = memoFounders?.map(founder => ({
    name: normaliseString(founder.name),
    phone:
      normaliseString((founder as unknown as { phone?: string }).phone) ||
      normaliseString((founder as unknown as { phone_number?: string }).phone_number) ||
      normaliseString((founder as unknown as { contact_number?: string }).contact_number),
  }));
  memoFounderPhones?.forEach(({ name, phone }) => {
    if (phone) {
      registerPhone(name || undefined, phone);
    }
  });

  phoneNumbers.forEach(({ displayName, phones }, nameKey) => {
    if (nameKey !== "__primary__" && displayName) {
      phones.forEach(phone => addPhoneToRecord(displayName, phone));
      return;
    }
    const firstRecord = directory.values().next().value as
      | { name: string; emails: Set<string>; phones: Set<string> }
      | undefined;
    if (firstRecord) {
      phones.forEach(phone => firstRecord.phones.add(phone));
    } else {
      const record = ensureRecord("Primary Contact");
      phones.forEach(phone => record.phones.add(phone));
    }
  });

  if (directory.size === 0) {
    ensureRecord("Primary Contact");
  }

  return Array.from(directory.values()).map((entry, index) => ({
    id: `${analysisData.deal_id}-${index}`,
    name: entry.name,
    companyName,
    emails: Array.from(entry.emails),
    phones: Array.from(entry.phones),
  }));
};

const formatTime = (value: string): string => {
  if (!value) return "";
  const [hours, minutes] = value.split(":");
  if (hours === undefined || minutes === undefined) return value;
  const date = new Date();
  date.setHours(Number(hours), Number(minutes));
  return date.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
};

export function StartupContactDirectory({
  analysisData,
  startupId,
  initialError = null,
}: StartupContactDirectoryProps) {
  const router = useRouter();
  const { toast } = useToast();
  const [selectedDate, setSelectedDate] = useState<Date | undefined>(undefined);
  const [selectedTime, setSelectedTime] = useState("");
  const [selectedFounderIds, setSelectedFounderIds] = useState<Set<string>>(new Set());
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const startOfToday = useMemo(() => {
    const now = new Date();
    return new Date(now.getFullYear(), now.getMonth(), now.getDate());
  }, []);

  const founders = useMemo(() => {
    if (!analysisData) return [] as FounderContact[];
    return buildFounderDirectory(analysisData);
  }, [analysisData]);

  const foundersWithEmails = useMemo(
    () => founders.filter(founder => founder.emails.length > 0),
    [founders],
  );

  const companyName = useMemo(() => {
    if (!analysisData) return "the company";
    return (
      normaliseString(analysisData.metadata?.display_name) ||
      normaliseString(analysisData.metadata?.company_name) ||
      normaliseString(analysisData.metadata?.company_legal_name) ||
      "the company"
    );
  }, [analysisData]);

  const handleToggleFounder = (id: string, checked: boolean | string) => {
    setSelectedFounderIds(prev => {
      const next = new Set(prev);
      if (checked) {
        next.add(id);
      } else {
        next.delete(id);
      }
      return next;
    });
  };

  const handleBackNavigation = () => {
    if (typeof window !== "undefined" && window.history.length > 1) {
      router.back();
    } else {
      router.push(`/startup/${startupId}`);
    }
  };

  const handleSchedule = () => {
    const selected = foundersWithEmails.filter(founder => selectedFounderIds.has(founder.id));
    if (selected.length === 0) {
      toast({
        variant: "destructive",
        title: "No contacts selected",
        description: "Please choose at least one founder with an email address.",
      });
      return;
    }

    if (!selectedDate || !selectedTime) {
      toast({
        variant: "destructive",
        title: "Missing meeting details",
        description: "Select both a date and time for the meeting.",
      });
      return;
    }

    const recipientEmails = selected.flatMap(founder => founder.emails);
    if (recipientEmails.length === 0) {
      toast({
        variant: "destructive",
        title: "No email addresses available",
        description: "The selected founders do not have email addresses to contact.",
      });
      return;
    }

    const formattedDate = format(selectedDate, "MMMM d, yyyy");
    const formattedTime = formatTime(selectedTime);

    const body = `Hi guys, Lets connect on ${formattedDate} at ${formattedTime} for 30 mins to discuss further about the product from your company ${companyName}. Thank you.`;
    const subject = `Meeting Request - ${companyName}`;

    const gmailUrl = `https://mail.google.com/mail/?view=cm&fs=1&to=${encodeURIComponent(
      recipientEmails.join(","),
    )}&su=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;

    window.open(gmailUrl, "_blank", "noopener,noreferrer");
    setIsDialogOpen(false);
    setSelectedFounderIds(new Set());
    setSelectedDate(undefined);
    setSelectedTime("");
  };

  const renderDirectory = () => {
    if (initialError) {
      return <p className="text-sm text-destructive">{initialError}</p>;
    }

    if (!analysisData) {
      return <p className="text-sm text-muted-foreground">Unable to load founder contact information.</p>;
    }

    if (founders.length === 0) {
      return <p className="text-sm text-muted-foreground">No founder contact information is currently available.</p>;
    }

    return (
      <div className="space-y-4">
        {founders.map(founder => (
          <div key={founder.id} className="rounded-lg border p-4 shadow-sm">
            <p className="text-base font-semibold">{founder.name}</p>
            <div className="mt-3 space-y-2 text-sm text-muted-foreground">
              <p className="flex items-center gap-2">
                <Phone className="h-4 w-4" />
                {founder.phones.length > 0 ? founder.phones.join(", ") : "Not available"}
              </p>
              <p className="flex items-center gap-2">
                <Mail className="h-4 w-4" />
                {founder.emails.length > 0 ? founder.emails.join(", ") : "Not available"}
              </p>
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <main className="container mx-auto px-4 py-6 md:px-6 md:py-10">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <Button variant="ghost" onClick={handleBackNavigation} className="flex items-center gap-2">
          <ArrowLeft className="h-4 w-4" /> Back
        </Button>
        {analysisData && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <UserRound className="h-4 w-4 text-primary" />
            {companyName}
          </div>
        )}
      </div>
      <div className="grid gap-6 lg:grid-cols-[2fr,1fr]">
        <Card className="border-primary/20">
          <CardHeader>
            <CardTitle className="text-2xl font-headline">Founder Contact Directory</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[60vh] pr-4">{renderDirectory()}</ScrollArea>
          </CardContent>
        </Card>
        <Card className="border-primary/20">
          <CardHeader>
            <CardTitle className="text-2xl font-headline">Meeting Scheduler</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <p className="text-sm text-muted-foreground">
              Coordinate follow-up conversations with founders directly from Pitch Lens.
            </p>
            <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
              <DialogTrigger asChild>
                <Button className="w-full" disabled={!!initialError || !analysisData}>
                  Schedule Meeting
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-[480px]">
                <DialogHeader>
                  <DialogTitle>Select Meeting Details</DialogTitle>
                  <DialogDescription>
                    Choose the founders you would like to meet and specify the date and time.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-5">
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium">Founders</h4>
                    <div className="max-h-40 space-y-2 overflow-y-auto pr-2">
                      {foundersWithEmails.length === 0 ? (
                        <p className="text-sm text-muted-foreground">
                          No email addresses available for scheduling at this time.
                        </p>
                      ) : (
                        foundersWithEmails.map(founder => {
                          const isChecked = selectedFounderIds.has(founder.id);
                          return (
                            <Label
                              key={founder.id}
                              className={cn(
                                "flex cursor-pointer items-start gap-3 rounded-md border p-3 text-sm transition hover:bg-accent",
                                isChecked ? "border-primary bg-primary/5" : "border-border",
                              )}
                            >
                              <Checkbox
                                checked={isChecked}
                                onCheckedChange={checked => handleToggleFounder(founder.id, !!checked)}
                              />
                              <div className="space-y-1">
                                <p className="font-medium text-foreground">{founder.name}</p>
                                <p className="text-xs text-muted-foreground">{founder.emails.join(", ")}</p>
                                <p className="text-xs text-muted-foreground">{founder.companyName}</p>
                              </div>
                            </Label>
                          );
                        })
                      )}
                    </div>
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="flex flex-col gap-2">
                      <Label className="text-sm font-medium">Date</Label>
                      <Popover>
                        <PopoverTrigger asChild>
                          <Button
                            variant="outline"
                            className={cn("justify-start text-left font-normal", !selectedDate && "text-muted-foreground")}
                          >
                            <CalendarIcon className="mr-2 h-4 w-4" />
                            {selectedDate ? format(selectedDate, "PPP") : "Pick a date"}
                          </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-auto p-0" align="start">
                          <Calendar
                            mode="single"
                            selected={selectedDate}
                            onSelect={setSelectedDate}
                            disabled={date => (date ? date < startOfToday : false)}
                          />
                        </PopoverContent>
                      </Popover>
                    </div>
                    <div className="flex flex-col gap-2">
                      <Label className="text-sm font-medium">Time</Label>
                      <Input type="time" value={selectedTime} onChange={event => setSelectedTime(event.target.value)} />
                    </div>
                  </div>
                </div>
                <DialogFooter>
                  <Button onClick={handleSchedule} disabled={foundersWithEmails.length === 0}>
                    Schedule
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}

export default StartupContactDirectory;
