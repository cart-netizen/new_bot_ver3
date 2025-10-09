import React from 'react';
import { cn } from '../../utils/helpers.ts';

export function Input({ className, ...props }: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        'w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg',
        'text-white placeholder:text-gray-500',
        'focus:outline-none focus:ring-2 focus:ring-primary',
        className
      )}
      {...props}
    />
  );
}