import React from "react";
import { FaShoppingCart } from "react-icons/fa";

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
        <h3 className="text-2xl text-center font-semibold text-orange-600 mb-10">
          Best Sellers <span role="img" aria-label="shoe">👟</span>
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {products.map((product) => (
            <div
              key={product.name}
              className="bg-white rounded-2xl shadow-md p-4 text-center"
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
      <button className="fixed bottom-6 right-6 bg-orange-500 hover:bg-orange-600 text-white font-semibold px-5 py-3 rounded-full shadow-lg flex items-center space-x-2 animate-chat-bounce">
        <img
          src="/frankie-avatar.png"
          alt="Frankie avatar"
          className="w-6 h-6 rounded-full"
        />
        <span className="flex items-center gap-1">
          Chat with Frankie
          <span className="animate-sparkle inline-block">✨</span>
        </span>
      </button>
    </div>
  );
}