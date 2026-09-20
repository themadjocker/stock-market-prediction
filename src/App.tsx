import React, { useState } from 'react';
import { Bell, Mail, Search, Settings, Home, Wallet, Newspaper, BarChart2, Users, PhoneCall } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { clsx } from 'clsx';

const stocks = [
  { id: 'AAPL', name: 'Apple', logo: 'https://images.unsplash.com/photo-1611186871348-b1ce696e52c9?w=50&h=50&fit=crop', price: 310.40, change: -1.10 },
  { id: 'META', name: 'Meta', logo: 'https://images.unsplash.com/photo-1611162616305-c69b3fa7fbe0?w=50&h=50&fit=crop', price: 140.45, change: -0.10 },
  { id: 'MSFT', name: 'Microsoft', logo: 'https://images.unsplash.com/photo-1642132652075-2b0036464813?w=50&h=50&fit=crop', price: 240.98, change: 0.85 },
  { id: 'GOOGL', name: 'Google', logo: 'https://images.unsplash.com/photo-1573804633927-bfcbcd909acd?w=50&h=50&fit=crop', price: 99.12, change: -0.04 }
];

const generateDummyData = () => {
  return Array.from({ length: 30 }, (_, i) => ({
    date: `2024-${String(i + 1).padStart(2, '0')}`,
    value: 150 + Math.random() * 50
  }));
};

function App() {
  const [selectedStock, setSelectedStock] = useState(stocks[0]);
  const [data] = useState(generateDummyData());

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex">
        {/* Sidebar */}
        <div className="w-64 bg-white h-screen p-6 flex flex-col fixed">
          <div className="flex items-center gap-2 mb-8">
            <BarChart2 className="w-8 h-8" />
            <span className="text-xl font-bold">GoStock</span>
          </div>

          <div className="bg-black text-white p-4 rounded-2xl mb-8">
            <div className="text-sm">Total Investment</div>
            <div className="text-2xl font-bold">$5380.90</div>
            <div className="text-green-400 text-sm">+18.10%</div>
          </div>

          <nav className="space-y-2 flex-1">
            <NavItem icon={Home} label="Home" />
            <NavItem icon={BarChart2} label="Dashboard" active />
            <NavItem icon={Wallet} label="Wallet" />
            <NavItem icon={Newspaper} label="News" />
            <NavItem icon={BarChart2} label="Stock & Fund" />
          </nav>

          <div className="space-y-2">
            <NavItem icon={Users} label="Our Community" />
            <NavItem icon={Settings} label="Settings" />
            <NavItem icon={PhoneCall} label="Contact Us" />
          </div>
        </div>

        {/* Main Content */}
        <div className="ml-64 flex-1 p-8">
          {/* Header */}
          <header className="flex justify-between items-center mb-8">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder='Press "⌘" to search for various stocks'
                className="pl-10 pr-4 py-2 bg-white rounded-xl w-96 focus:outline-none"
              />
            </div>
            
            <div className="flex items-center gap-4">
              <button className="p-2 hover:bg-gray-100 rounded-xl">
                <Mail className="w-6 h-6" />
              </button>
              <button className="p-2 hover:bg-gray-100 rounded-xl">
                <Bell className="w-6 h-6" />
              </button>
              <div className="flex items-center gap-2">
                <img
                  src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=32&h=32&fit=crop&crop=faces"
                  alt="Profile"
                  className="w-8 h-8 rounded-full"
                />
                <span className="font-medium">John Doe</span>
              </div>
            </div>
          </header>

          {/* Stock Cards */}
          <div className="grid grid-cols-4 gap-4 mb-8">
            {stocks.map((stock) => (
              <button
                key={stock.id}
                onClick={() => setSelectedStock(stock)}
                className={clsx(
                  "bg-white p-4 rounded-2xl transition-all",
                  selectedStock.id === stock.id ? "ring-2 ring-blue-500" : ""
                )}
              >
                <div className="flex items-center gap-3 mb-2">
                  <img src={stock.logo} alt={stock.name} className="w-8 h-8 rounded-full" />
                  <span className="font-medium">{stock.name}</span>
                </div>
                <div className="text-xl font-bold">${stock.price}</div>
                <div className={clsx(
                  "text-sm",
                  stock.change > 0 ? "text-green-500" : "text-red-500"
                )}>
                  {stock.change > 0 ? "+" : ""}{stock.change}%
                </div>
              </button>
            ))}
          </div>

          {/* Stock Graph */}
          <div className="bg-white p-6 rounded-2xl">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-2xl font-bold">{selectedStock.name}</h2>
                <p className="text-gray-500">{selectedStock.id}</p>
              </div>
              <div className="flex gap-2">
                {['1D', '1W', '1M', '3M', '6M', '1Y', '5Y', 'All'].map((period) => (
                  <button
                    key={period}
                    className="px-4 py-1 rounded-full text-sm hover:bg-gray-100"
                  >
                    {period}
                  </button>
                ))}
              </div>
            </div>
            
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#2563eb"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function NavItem({ icon: Icon, label, active }: { icon: any, label: string, active?: boolean }) {
  return (
    <button
      className={clsx(
        "flex items-center gap-3 w-full p-3 rounded-xl transition-colors",
        active ? "bg-gray-100" : "hover:bg-gray-50"
      )}
    >
      <Icon className="w-5 h-5" />
      <span>{label}</span>
    </button>
  );
}

export default App;