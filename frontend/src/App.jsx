import React, { useState, useEffect, useRef } from "react";
import { FaShoppingCart } from "react-icons/fa";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const products = [
  {
    name: "Breeze Runners",
    price: "$79.00",
    image: "/breeze-runners.jpg",
  },
  {
    name: "Cloud Walkers",
    price: "$74.00",
    image: "/cloud-walkers.jpg",
  },
  {
    name: "Sunset Sneakers",
    price: "$69.00",
    image: "/white-sneakers.jpg",
  },
];

export default function App() {
  const [showChat, setShowChat] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hi, I'm Frankie! How can I help you today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      console.log("📨 Sending to backend:", input);

      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: input,
          history: messages,
         }),
      });

      const data = await res.json();

      console.log("📥 Received from backend:", data);

      const botMessage = { role: "assistant", content: data.response };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error("❌ Error in fetch:", err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Something went wrong contacting the server." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const toggleChat = () => {
    if (showChat) {
      setMessages([
        {
          role: "assistant",
          content: "Hi, I'm Frankie! How can I help you today?",
        },
      ]);
    }
    setShowChat(!showChat);
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="min-h-screen bg-orange-50 font-sans relative">
      {/* Navbar */}
      <header className="flex justify-between items-center px-6 py-4 bg-white shadow">
        <h1 className="text-2xl font-extrabold text-orange-600">HappyFeet</h1>
        <nav className="space-x-6 text-gray-800 font-medium flex items-center">
          <a href="#" className="hover:text-orange-600">Home</a>
          <a href="#" className="hover:text-orange-600">Shop</a>
          <a href="#" className="hover:text-orange-600">About</a>
          <a href="#" className="hover:text-orange-600 flex items-center">
            Cart <FaShoppingCart className="ml-1" />
          </a>
        </nav>
      </header>

      {/* Hero */}
      <section className="bg-orange-100 text-center py-10 px-6">
        <h2 className="text-4xl font-extrabold text-orange-600 mb-3">
          Step Into Something Comfy
        </h2>
        <p className="text-gray-700 text-lg max-w-xl mx-auto">
          Shoes that will bring you joy on your journey. Discover fun, functional footwear designed for comfort and joy — only at HappyFeet.
        </p>
      </section>

      {/* Product Grid */}
      <section className="py-10 px-6">
        <h3 className="text-2xl text-center font-semibold text-orange-600 mb-10 flex justify-center items-center gap-2">
          <span className="animate-bounce">🔥</span>
          Best Sellers
          <span className="animate-bounce">🔥</span>
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {products.map((product) => (
            <div
              key={product.name}
              onClick={() => alert(`You selected: ${product.name}`)}
              className="bg-white rounded-2xl shadow-md p-4 text-center cursor-pointer transform transition-transform duration-300 hover:scale-105"
            >
              <img
                src={product.image}
                alt={product.name}
                className="mx-auto h-48 object-contain mb-4"
              />
              <h4 className="text-lg font-semibold">{product.name}</h4>
              <p className="text-orange-600 font-semibold mt-2">{product.price}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Chat Button */}
      <button
        onClick={toggleChat}
        className="fixed bottom-6 right-6 bg-orange-500 hover:bg-orange-600 text-white font-semibold px-5 py-3 rounded-full shadow-lg flex items-center space-x-2 animate-chat-bounce"
      >
        <img
          src="/frankie-avatar.png"
          alt="Frankie avatar"
          className="w-6 h-6 rounded-full"
        />
        <span className="flex items-center gap-1">
          Chat with Frankie <span className="animate-ping inline-block">✨</span>
        </span>
      </button>

      {/* Chat Box */}
      {showChat && (
        <div className="fixed bottom-24 right-6 w-80 h-96 bg-white border shadow-xl rounded-2xl flex flex-col p-4 z-50 animate-fade-in">
          <div className="flex items-center gap-2 mb-2 font-bold text-orange-600">
            <img
              src="/frankie-avatar.png"
              alt="Frankie avatar"
              className="w-6 h-6 rounded-full"
            />
            Frankie
          </div>
          <div className="flex-1 overflow-y-auto text-sm text-gray-700 space-y-2 mb-2">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`whitespace-pre-wrap px-4 py-2 rounded-2xl break-words text-center ${
                  msg.role === "user"
                    ? "bg-orange-100 ml-auto max-w-[80%]"
                    : "bg-gray-100 text-left mr-auto max-w-[85%]"
                }`}
              >
                {typeof msg.content === "string" && msg.content.trim() && (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      a: ({ node, ...props }) => (
                        <a
                          {...props}
                          className="text-orange-600 underline hover:text-orange-800"
                          target="_blank"
                          rel="noopener noreferrer"
                        />
                      ),
                      p: ({ node, ...props }) => (
                        <p className="my-1 leading-snug" {...props} />
                      ),
                      li: ({ node, ...props }) => (
                        <li className="my-1 leading-snug" {...props} />
                      ),
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                )}
              </div>
            ))}
            {loading && (
              <div className="p-2 bg-gray-100 mr-8 rounded-md italic text-gray-500 animate-pulse">
                Frankie is typing...
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <form onSubmit={handleSubmit} className="mt-auto">
          <textarea
            rows={1}
            placeholder="Type your question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
            className="w-full resize-none px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-orange-300"
          />
          </form>
        </div>
      )}
    </div>
  );
}