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
  scriptedReplies?: string[];
}

export const examplePrompts: string[] = [
  "How do I reset my password without losing my data?",
  "My order shows delivered but I can't find it. What should I do?",
  "I want to cancel my subscription before it renews.",
  "The checkout page is stuck loading. What steps should I try?",
  "Where can I find your privacy policy and terms?"
];

import trainingThreads from "./training_threads.json";

const normalizeThreads = (threads: Thread[]): Thread[] => {
  const normalizeMessages = (messages: Message[]) => {
    const filtered = messages.filter((message) => message.role !== "system");
    const trimmed = [...filtered];
    while (trimmed.length && trimmed[0].role !== "user") {
      trimmed.shift();
    }
    return trimmed;
  };

  return threads.map((thread, tIndex) => ({
    ...thread,
    id: thread.id || `thread-${tIndex + 1}`,
    title: thread.title || "Conversation",
    messages: normalizeMessages(thread.messages).map((message, mIndex) => ({
      ...message,
      id: message.id || `${thread.id}-m${mIndex + 1}`,
      timestamp: message.timestamp || ""
    }))
  }));
};

export const initialThreads: Thread[] = normalizeThreads(
  (trainingThreads as Thread[]).length ? (trainingThreads as Thread[]) : [
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
      },
      {
        id: "m-3",
        role: "user",
        content: "I already checked with neighbors. The carrier says it's delivered.",
        timestamp: "Today 10:16 AM"
      },
      {
        id: "m-4",
        role: "assistant",
        content:
          "If the carrier confirms delivery, file a missing package claim and request the delivery scan and location details. Keep your order confirmation and tracking number handy. You can also check if your building has a concierge or mailroom log. If the claim confirms a loss, submit the claim outcome in your order history to proceed with a replacement or refund.",
        timestamp: "Today 10:17 AM"
      }
    ],
    scriptedReplies: [
      "If you'd like, share the tracking number and I can outline the exact steps to submit the carrier claim and where to upload the confirmation in your order history."
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
      },
      {
        id: "m-5",
        role: "user",
        content: "I don't see the cancel button.",
        timestamp: "Yesterday 4:30 PM"
      },
      {
        id: "m-6",
        role: "assistant",
        content:
          "If the cancel button is missing, check that you're in the correct account and that the subscription is active. Try switching to the web dashboard if you're in the mobile app. Also look under Billing → Plans for the cancellation option. If it's still not visible, your plan may be managed through a third‑party app store, which means cancellation needs to happen in that store's subscription settings.",
        timestamp: "Yesterday 4:31 PM"
      }
    ],
    scriptedReplies: [
      "If you confirm where you subscribed (web, App Store, or Google Play), I can give the exact cancellation path."
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
      },
      {
        id: "m-7",
        role: "user",
        content: "Tried incognito and a different card. Still reloading.",
        timestamp: "Mon 2:09 PM"
      },
      {
        id: "m-8",
        role: "assistant",
        content:
          "Next, try a different browser or device and make sure your network isn't blocking payment scripts. If the problem persists, capture the exact error from the browser console or any on‑screen message. You can also try removing items and adding them back to your cart to clear any stale checkout state.",
        timestamp: "Mon 2:10 PM"
      }
    ],
    scriptedReplies: [
      "If you can share the browser and device you're using, I can provide a targeted troubleshooting checklist."
    ]
  },
  {
    id: "thread-4",
    title: "Refund Status Check",
    messages: [
      {
        id: "m-9",
        role: "user",
        content: "I returned an item last week. Where is my refund?",
        timestamp: "Sun 11:14 AM"
      },
      {
        id: "m-10",
        role: "assistant",
        content:
          "Go to Returns & Refunds in your account to see the status. Once the return is marked received, refunds are processed to your original payment method. You can also check your bank statement for the credit.",
        timestamp: "Sun 11:15 AM"
      },
      {
        id: "m-11",
        role: "user",
        content: "It says received but no refund yet.",
        timestamp: "Sun 11:17 AM"
      },
      {
        id: "m-12",
        role: "assistant",
        content:
          "If the return is marked received, allow a few business days for processing. Some banks take extra time to post credits. Keep an eye on your statement and the return status page. If it stays unchanged past the normal processing window, you can use the return details page to request a refund review.",
        timestamp: "Sun 11:18 AM"
      }
    ],
    scriptedReplies: [
      "If you share the return ID, I can point you to the exact status page and where to request a review."
    ]
  }
]
);
