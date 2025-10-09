import { create } from 'zustand';
import { botApi } from '../api/bot.api.ts';
import type { ConfigResponse } from '../types/api.types.ts';
// import {BotStatus} from "../types/botStatus.ts";
import {BotStatus} from "../types/common.types.ts";

interface BotState {
  status: BotStatus;
  config: ConfigResponse | null;
  symbols: string[];
  isLoading: boolean;
  startBot: () => Promise<void>;
  stopBot: () => Promise<void>;
  fetchStatus: () => Promise<void>;
  fetchConfig: () => Promise<void>;
  updateStatus: (status: BotStatus) => void;
}

export const useBotStore = create<BotState>((set) => ({
  status: BotStatus.STOPPED,
  config: null,
  symbols: [],
  isLoading: false,

  startBot: async () => {
    set({ isLoading: true });
    try {
      await botApi.start();
      set({ status: BotStatus.STARTING, isLoading: false });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  stopBot: async () => {
    set({ isLoading: true });
    try {
      await botApi.stop();
      set({ status: BotStatus.STOPPING, isLoading: false });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  fetchStatus: async () => {
    try {
      const data = await botApi.getStatus();
      set({
        status: data.status,
        symbols: data.symbols || [],
      });
    } catch (error) {
      console.error('Failed to fetch status:', error);
    }
  },

  fetchConfig: async () => {
    try {
      const config = await botApi.getConfig();
      set({ config });
    } catch (error) {
      console.error('Failed to fetch config:', error);
    }
  },

  updateStatus: (status) => set({ status }),
}));