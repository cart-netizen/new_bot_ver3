import React from 'react';
import { cn } from '../../utils/helpers.ts';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {}

export function Card({ className, ...props }: CardProps) {
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