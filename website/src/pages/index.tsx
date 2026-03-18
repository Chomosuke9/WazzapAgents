import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/memulai">
            Mulai Sekarang
          </Link>
          <Link
            className="button button--outline button--lg"
            to="/perintah"
            style={{color: '#fff', borderColor: '#fff'}}>
            Lihat Perintah
          </Link>
        </div>
      </div>
    </header>
  );
}

function Feature({title, description, icon}: {title: string; description: string; icon: string}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="feature-card" style={{textAlign: 'center', paddingTop: '1.5rem', paddingBottom: '1.5rem'}}>
        <div style={{fontSize: '2.5rem', marginBottom: '0.75rem'}}>{icon}</div>
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

function HowItWorks() {
  const steps = [
    { num: '1', title: 'Tambahkan Bot', desc: 'Masukkan nomor bot ke grup WhatsApp-mu seperti menambah anggota biasa.' },
    { num: '2', title: 'Atur Prompt', desc: 'Tentukan kepribadian dan aturan bot dengan perintah /prompt.' },
    { num: '3', title: 'Siap Digunakan!', desc: 'Bot langsung aktif dan bisa diajak ngobrol oleh semua anggota grup.' },
  ];

  return (
    <section className="how-it-works">
      <div className="container">
        <Heading as="h2" style={{textAlign: 'center', marginBottom: '2rem'}}>
          Cara Kerja
        </Heading>
        <div className="row" style={{justifyContent: 'center'}}>
          {steps.map((step, idx) => (
            <div key={idx} className="col col--3" style={{textAlign: 'center', padding: '1rem'}}>
              <div className="step-number">{step.num}</div>
              <Heading as="h4">{step.title}</Heading>
              <p style={{fontSize: '0.95rem'}}>{step.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

const features = [
  {
    title: 'Asisten AI di WhatsApp',
    description: 'Bot berbasis AI yang bisa mengobrol natural, menjawab pertanyaan, bercanda, dan membantu anggota grup.',
    icon: '🤖',
  },
  {
    title: 'Moderasi Otomatis',
    description: 'Bisa menghapus pesan spam atau mengeluarkan anggota nakal secara otomatis sesuai aturan yang kamu set.',
    icon: '🛡️',
  },
  {
    title: 'Sepenuhnya Bisa Dikustomisasi',
    description: 'Atur kepribadian, peran, dan aturan bot dengan perintah /prompt. Berbeda di setiap grup.',
    icon: '🎨',
  },
];

export default function Home(): ReactNode {
  return (
    <Layout
      title="Panduan Pengguna"
      description="Dokumentasi lengkap cara menggunakan WazzapAgents — bot WhatsApp berbasis AI">
      <HomepageHeader />
      <main>
        <section style={{padding: '3rem 0'}}>
          <div className="container">
            <div className="row">
              {features.map((props, idx) => (
                <Feature key={idx} {...props} />
              ))}
            </div>
          </div>
        </section>
        <HowItWorks />
      </main>
    </Layout>
  );
}
