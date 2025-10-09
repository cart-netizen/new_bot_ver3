// src/components/ui/Button.tsx
import { type ReactNode, type ButtonHTMLAttributes } from 'react';
import { cn } from '@/utils/helpers';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'destructive' | 'outline';
  className?: string;
  children?: ReactNode;
}

export const Button = ({
  className,
  variant = 'default',
  children,
  disabled,
  ...props
}: ButtonProps) => {
  return (
    <button
      disabled={disabled}
      className={cn(
        'inline-flex items-center justify-center rounded-lg px-4 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
        'disabled:pointer-events-none',
        variant === 'default' &&
          'bg-primary text-primary-foreground hover:bg-primary/90',
        variant === 'destructive' &&
          'bg-destructive text-destructive-foreground hover:bg-destructive/90',
        variant === 'outline' &&
          'border border-input bg-background hover:bg-accent hover:text-accent-foreground',
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
};