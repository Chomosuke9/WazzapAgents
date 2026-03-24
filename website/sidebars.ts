import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'index',
      label: 'Pengantar',
    },
    {
      type: 'doc',
      id: 'memulai',
      label: 'Cara Memulai',
    },
    {
      type: 'category',
      label: 'Perintah Bot',
      items: [
        'perintah',
        'permission',
      ],
    },
    {
      type: 'doc',
      id: 'prompt',
      label: 'Mengatur Prompt',
    },
    {
      type: 'doc',
      id: 'contoh-prompt',
      label: 'Contoh Prompt Siap Pakai',
    },
    {
      type: 'doc',
      id: 'fitur',
      label: 'Fitur-Fitur Bot',
    },
    {
      type: 'doc',
      id: 'tips',
      label: 'Tips & Praktik Terbaik',
    },
    {
      type: 'doc',
      id: 'faq',
      label: 'FAQ',
    },
    {
      type: 'category',
      label: 'Dokumentasi Developer',
      items: [
        'dev/arsitektur',
        'dev/setup',
        'dev/gateway',
        'dev/bridge',
        'dev/protokol',
        'dev/kontribusi',
      ],
    },
  ],
};

export default sidebars;
