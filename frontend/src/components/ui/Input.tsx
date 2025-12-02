import { cn } from '../../utils/helpers';
import React from "react";

interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'className'> {
  className?: string;
}

export function Input({ className, ...props }: InputProps) {
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