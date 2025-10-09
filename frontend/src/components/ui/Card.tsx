import React from 'react';
import { cn } from '../../utils/helpers.ts';

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        'bg-surface rounded-lg border border-gray-800',
        className
      )}
      {...props}
    />
  );
}