import {ReactNode} from 'react';
import { cn } from '../../utils/helpers.ts';

interface CardProps {
  className?: string;
  children?: ReactNode;
  onClick?: () => void;
}

export function Card({ className, children, ...props }: CardProps) {
  return (
    <div
      className={cn(
        'bg-surface rounded-lg border border-gray-800',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}