export const BotStatus = {
  STOPPED: 'STOPPED',
  STARTING: 'STARTING',
  RUNNING: 'RUNNING',
  STOPPING: 'STOPPING',
  ERROR: 'ERROR',
} as const;

export type BotStatus = typeof BotStatus[keyof typeof BotStatus];