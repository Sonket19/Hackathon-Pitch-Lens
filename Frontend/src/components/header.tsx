import Link from 'next/link';
import { BrainCircuit } from 'lucide-react';
import Image from 'next/image';

import { cn } from '@/lib/utils';
import { buttonVariants } from '@/components/ui/button';
import Pitch from '../app/Pitch.png';

export default function Header() {
  return (
    <header
      className={cn(
        'sticky top-0 z-50 border-b bg-card/50 backdrop-blur-sm',
        'px-4 py-4 md:px-6'
      )}
    >
      <div className="container mx-auto flex items-center justify-between">
        <Link
          href="/"
          className={cn(
            buttonVariants({ variant: 'ghost', size: 'lg' }),
            'h-auto px-0 py-0 text-base font-semibold text-foreground',
            'flex items-center gap-3'
          )}
        >
          {/* <div className="bg-primary p-2 rounded-lg">
            <BrainCircuit className="w-6 h-6 text-primary-foreground" />
          </div> */}
          <Image src={Pitch} alt="Pitch Logo" width={100} height={100} />
          <h1 className="text-2xl md:text-3xl font-headline font-bold text-foreground">
            Pitch Lens
          </h1>
        </Link>
        <Link
          href="/contact"
          className={cn(buttonVariants({ variant: 'secondary' }), 'font-semibold')}
        >
          Contact
        </Link>
      </div>
    </header>
  );
}
