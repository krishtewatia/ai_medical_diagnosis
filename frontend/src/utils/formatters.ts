/**
 * Utility formatting functions for dates, percentages, and numbers.
 */

export function formatDate(dateString: string): string {
  try {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  } catch {
    return dateString;
  }
}

export function formatProbability(value?: number): string {
  if (value === undefined || value === null || isNaN(value)) {
    return "N/A";
  }
  return `${(value * 100).toFixed(1)}%`;
}

export function formatFeatureName(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/^./, (str) => str.toUpperCase());
}
