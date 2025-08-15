import React from 'react';
import { 
  MessageSquare, 
  FileText, 
  Youtube, 
  Brain, 
  BarChart3,
  Sparkles
} from 'lucide-react';

const Navigation = ({ activeTab, onTabChange }) => {
  const tabs = [
    {
      id: 'single',
      name: 'Single Analysis',
      icon: MessageSquare,
      description: 'Analyze individual text'
    },
    {
      id: 'batch',
      name: 'Batch Analysis',
      icon: FileText,
      description: 'Process multiple texts'
    },
    {
      id: 'youtube',
      name: 'YouTube Analysis',
      icon: Youtube,
      description: 'Analyze video comments'
    },
    {
      id: 'training',
      name: 'Model Training',
      icon: Brain,
      description: 'Train custom models'
    },
    {
      id: 'dashboard',
      name: 'Dashboard',
      icon: BarChart3,
      description: 'Analytics & insights'
    }
  ];

  return (
    <nav className="bg-white rounded-xl shadow-lg border border-gray-100 p-2">
      <div className="flex flex-wrap gap-2">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`
                flex items-center space-x-2 px-4 py-3 rounded-lg font-medium transition-all duration-200
                ${isActive 
                  ? 'bg-blue-600 text-white shadow-md transform scale-105' 
                  : 'text-gray-600 hover:text-blue-600 hover:bg-blue-50'
                }
              `}
              title={tab.description}
            >
              <Icon className={`h-5 w-5 ${isActive ? 'text-white' : ''}`} />
              <span className="hidden sm:inline">{tab.name}</span>
              {isActive && (
                <Sparkles className="h-4 w-4 text-blue-200 animate-pulse" />
              )}
            </button>
          );
        })}
      </div>
    </nav>
  );
};

export default Navigation;
