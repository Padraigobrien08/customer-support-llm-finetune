export type MessageRole = "user" | "assistant";

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: string;
}

export interface Thread {
  id: string;
  title: string;
  messages: Message[];
}

export const examplePrompts: string[] = [
  "How do I reset my password without losing my data?",
  "My order shows delivered but I can't find it. What should I do?",
  "I want to cancel my subscription before it renews.",
  "The checkout page is stuck loading. What steps should I try?",
  "Where can I find your privacy policy and terms?"
];

export const initialThreads: Thread[] = [
  {
    id: "thread-1",
    title: "Order Delivery Follow-up",
    messages: [
      {
        id: "m-1",
        role: "user",
        content: "My order says delivered but it's not here.",
        timestamp: "Today 10:12 AM"
      },
      {
        id: "m-2",
        role: "assistant",
        content:
          "Check around your property, mailroom, and with neighbors. If it's still missing, contact the carrier using the tracking number for delivery details. You can also file a claim with the carrier if needed.",
        timestamp: "Today 10:13 AM"
      }
    ]
  },
  {
    id: "thread-2",
    title: "Subscription Cancellation",
    messages: [
      {
        id: "m-3",
        role: "user",
        content: "I want to cancel my subscription before renewal.",
        timestamp: "Yesterday 4:28 PM"
      },
      {
        id: "m-4",
        role: "assistant",
        content:
          "Go to Account Settings → Subscriptions, click Cancel Subscription, and confirm. You will keep access until the end of your billing period.",
        timestamp: "Yesterday 4:29 PM"
      }
    ]
  },
  {
    id: "thread-3",
    title: "Troubleshooting Checkout",
    messages: [
      {
        id: "m-5",
        role: "user",
        content: "Checkout keeps reloading when I submit payment.",
        timestamp: "Mon 2:05 PM"
      },
      {
        id: "m-6",
        role: "assistant",
        content:
          "Clear cache and cookies, disable extensions, try a private window, and verify all required fields are filled. If autofill is on, try manual entry.",
        timestamp: "Mon 2:06 PM"
      }
    ]
  }
];
