export type ClassValue = string | number | boolean | undefined | null | { [key: string]: unknown } | ClassValue[];

export function cn(...inputs: ClassValue[]): string {
  const classes: string[] = [];

  function process(input: ClassValue) {
    if (!input) return;
    if (typeof input === 'string' || typeof input === 'number') {
      classes.push(String(input));
    } else if (Array.isArray(input)) {
      for (const item of input) {
        process(item);
      }
    } else if (typeof input === 'object') {
      for (const key of Object.keys(input)) {
        if (input[key]) {
          classes.push(key);
        }
      }
    }
  }

  for (const input of inputs) {
    process(input);
  }

  return classes.join(' ');
}
