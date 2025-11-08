// frontend/src/components/backtesting/TemplateLibrary.tsx

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Save,
  Folder,
  Trash2,

  Copy,
  Tag,
  Clock,
  TrendingUp,
  Search,
  X,

} from 'lucide-react';
import { toast } from 'sonner';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { cn } from '../../utils/helpers';
import * as templatesApi from '../../api/templates.api';
import type { Template, TemplateConfig, CreateTemplateRequest } from '../../api/templates.api';

interface ApiError {
  response?: {
    data?: {
      detail?: string;
    };
  };
}

interface TemplateLibraryProps {
  onLoadTemplate: (config: TemplateConfig) => void;
  currentConfig?: TemplateConfig;
}

export function TemplateLibrary({ onLoadTemplate, currentConfig }: TemplateLibraryProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const queryClient = useQueryClient();

  // Fetch templates
  const { data: templatesData, isLoading } = useQuery({
    queryKey: ['templates', selectedTags],
    queryFn: () => templatesApi.listTemplates(selectedTags)
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: templatesApi.deleteTemplate,
    onSuccess: () => {
      toast.success('Шаблон удален');
      queryClient.invalidateQueries({ queryKey: ['templates'] });
    },
    onError: (error: ApiError) => {
      toast.error(error.response?.data?.detail || 'Ошибка удаления шаблона');
    }
  });

  const templates = templatesData?.templates || [];

  // Filter by search term
  const filteredTemplates = templates.filter((template: Template) =>
    template.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    template.description?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Get all unique tags
  const allTags = Array.from(
    new Set(templates.flatMap((t: Template) => t.tags || []))
  ) as string[];

  const toggleTag = (tag: string) => {
    setSelectedTags(prev =>
      prev.includes(tag)
        ? prev.filter(t => t !== tag)
        : [...prev, tag]
    );
  };

  const handleLoadTemplate = (template: Template) => {
    onLoadTemplate(template.config);
    toast.success(`Шаблон "${template.name}" загружен`);
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Folder className="h-6 w-6 text-purple-400" />
            <h3 className="text-lg font-semibold text-white">Библиотека шаблонов</h3>
          </div>
          <Button
            onClick={() => setShowSaveDialog(true)}
            disabled={!currentConfig}
            className="px-3 py-1.5 text-sm"
          >
            <Save className="h-4 w-4 mr-2" />
            Сохранить текущий
          </Button>
        </div>

        {/* Search */}
        <div className="relative mb-4">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Поиск шаблонов..."
            className="w-full pl-10 pr-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="absolute right-3 top-1/2 transform -translate-y-1/2"
            >
              <X className="h-4 w-4 text-gray-400 hover:text-white" />
            </button>
          )}
        </div>

        {/* Tags Filter */}
        {allTags.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {allTags.map((tag) => (
              <button
                key={tag}
                onClick={() => toggleTag(tag)}
                className={cn(
                  "flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium transition-colors",
                  selectedTags.includes(tag)
                    ? "bg-blue-500/20 text-blue-300 border border-blue-500/50"
                    : "bg-gray-800 text-gray-400 border border-gray-700 hover:border-gray-600"
                )}
              >
                <Tag className="h-3 w-3" />
                {tag}
              </button>
            ))}
          </div>
        )}
      </Card>

      {/* Templates Grid */}
      {isLoading ? (
        <Card className="p-8 text-center">
          <p className="text-gray-400">Загрузка шаблонов...</p>
        </Card>
      ) : filteredTemplates.length === 0 ? (
        <Card className="p-8 text-center">
          <Folder className="h-12 w-12 text-gray-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-white mb-2">Нет шаблонов</h3>
          <p className="text-gray-400">
            {searchTerm || selectedTags.length > 0
              ? 'По вашему запросу ничего не найдено'
              : 'Создайте свой первый шаблон для быстрого доступа'}
          </p>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredTemplates.map((template: Template) => (
            <TemplateCard
              key={template.id}
              template={template}
              onLoad={handleLoadTemplate}
              onDelete={(id) => deleteMutation.mutate(id)}
            />
          ))}
        </div>
      )}

      {/* Save Dialog */}
      {showSaveDialog && currentConfig && (
        <SaveTemplateDialog
          config={currentConfig}
          onClose={() => setShowSaveDialog(false)}
          onSaved={() => {
            setShowSaveDialog(false);
            queryClient.invalidateQueries({ queryKey: ['templates'] });
          }}
        />
      )}
    </div>
  );
}

// Template Card Component
interface TemplateCardProps {
  template: Template;
  onLoad: (template: Template) => void;
  onDelete: (id: string) => void;
}

function TemplateCard({ template, onLoad, onDelete }: TemplateCardProps) {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  return (
    <Card className="p-4 hover:border-gray-700 transition-all">
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h4 className="text-white font-semibold mb-1">{template.name}</h4>
          {template.description && (
            <p className="text-sm text-gray-400 line-clamp-2">{template.description}</p>
          )}
        </div>
        {template.is_public && (
          <span className="px-2 py-0.5 text-xs font-medium rounded bg-green-500/20 text-green-300">
            Public
          </span>
        )}
      </div>

      {/* Tags */}
      {template.tags && template.tags.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-3">
          {template.tags.map((tag: string) => (
            <span
              key={tag}
              className="px-2 py-0.5 text-xs font-medium rounded bg-gray-800 text-gray-300"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Meta */}
      <div className="flex items-center gap-4 text-xs text-gray-500 mb-3">
        <div className="flex items-center gap-1">
          <Clock className="h-3 w-3" />
          <span>{new Date(template.created_at).toLocaleDateString('ru-RU')}</span>
        </div>
        <div className="flex items-center gap-1">
          <TrendingUp className="h-3 w-3" />
          <span>{template.usage_count} раз</span>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <Button
          onClick={() => onLoad(template)}
          className="flex-1 px-3 py-1.5 text-sm"
        >
          <Copy className="h-3 w-3 mr-1" />
          Загрузить
        </Button>
        {!showDeleteConfirm ? (
          <Button
            onClick={() => setShowDeleteConfirm(true)}
            variant="outline"
            className="px-3 py-1.5 text-sm"
          >
            <Trash2 className="h-3 w-3" />
          </Button>
        ) : (
          <Button
            onClick={() => {
              onDelete(template.id);
              setShowDeleteConfirm(false);
            }}
            variant="outline"
            className="bg-red-500/20 border-red-500/50 text-red-300 px-3 py-1.5 text-sm"
          >
            Удалить?
          </Button>
        )}
      </div>
    </Card>
  );
}

// Save Template Dialog
interface SaveTemplateDialogProps {
  config: TemplateConfig;
  onClose: () => void;
  onSaved: () => void;
}

function SaveTemplateDialog({ config, onClose, onSaved }: SaveTemplateDialogProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState('');
  const [isPublic, setIsPublic] = useState(false);

  const saveMutation = useMutation({
    mutationFn: (data: CreateTemplateRequest) => templatesApi.createTemplate(data),
    onSuccess: () => {
      toast.success('Шаблон сохранен');
      onSaved();
    },
    onError: (error: ApiError) => {
      toast.error(error.response?.data?.detail || 'Ошибка сохранения шаблона');
    }
  });

  const handleSave = () => {
    if (!name.trim()) {
      toast.error('Введите название шаблона');
      return;
    }

    const tagList = tags
      .split(',')
      .map(t => t.trim())
      .filter(t => t.length > 0);

    saveMutation.mutate({
      name: name.trim(),
      description: description.trim() || undefined,
      config,
      tags: tagList,
      is_public: isPublic
    });
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <Card className="w-full max-w-md p-6 m-4">
        <h3 className="text-lg font-semibold text-white mb-4">Сохранить шаблон</h3>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Название *
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Например: Conservative Scalping"
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Описание
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Краткое описание стратегии..."
              rows={3}
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Теги (через запятую)
            </label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="scalping, conservative, btc"
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="flex items-center gap-3">
            <input
              type="checkbox"
              id="is_public"
              checked={isPublic}
              onChange={(e) => setIsPublic(e.target.checked)}
              className="h-4 w-4 rounded border-gray-700 bg-gray-900 text-blue-500 focus:ring-2 focus:ring-blue-500"
            />
            <label htmlFor="is_public" className="text-sm text-gray-300">
              Сделать публичным (доступен всем пользователям)
            </label>
          </div>
        </div>

        <div className="flex gap-3 mt-6">
          <Button onClick={onClose} variant="outline" className="flex-1">
            Отмена
          </Button>
          <Button
            onClick={handleSave}
            disabled={saveMutation.isPending}
            className="flex-1"
          >
            {saveMutation.isPending ? 'Сохранение...' : 'Сохранить'}
          </Button>
        </div>
      </Card>
    </div>
  );
}
