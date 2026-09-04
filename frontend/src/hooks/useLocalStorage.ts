'use client';

import { useCallback, useSyncExternalStore } from 'react';

const listeners = new Set<() => void>();

function subscribe(onChange: () => void) {
  listeners.add(onChange);
  window.addEventListener('storage', onChange);
  return () => {
    listeners.delete(onChange);
    window.removeEventListener('storage', onChange);
  };
}

function emit() {
  listeners.forEach((l) => l());
}

/**
 * localStorage as an external store, so preferences read correctly on the
 * client without a setState-in-effect hydration dance.
 */
export function useStoredValue(key: string, fallback: string) {
  const value = useSyncExternalStore(
    subscribe,
    () => window.localStorage.getItem(key) ?? fallback,
    () => fallback,
  );

  const setValue = useCallback(
    (next: string) => {
      window.localStorage.setItem(key, next);
      emit();
    },
    [key],
  );

  return [value, setValue] as const;
}

export function useStoredFlag(key: string, fallback = false) {
  const [raw, setRaw] = useStoredValue(key, fallback ? '1' : '0');
  const setFlag = useCallback((next: boolean) => setRaw(next ? '1' : '0'), [setRaw]);
  return [raw === '1', setFlag] as const;
}
