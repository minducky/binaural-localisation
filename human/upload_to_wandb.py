import re
import pandas as pd
import wandb

CSV_PATH = ('saddler_2023_snr.csv')

METRIC_COLUMNS = {
    'spherical_error':    'deg_err',
    'azimuth_error':      'deg_azim_err',
    'elevation_error':    'deg_elev_err',
    'combined_accuracy':  'correct',
    'azimuth_accuracy':   'correct_azim',
    'elevation_accuracy': 'correct_elev',
}


def extract_subject(fn_eval):
    match = re.search(r'subject(\w+)_', fn_eval)
    return match.group(1) if match else fn_eval


def main():
    df = pd.read_csv(CSV_PATH)
    df['snr'] = pd.to_numeric(df['snr'], errors='coerce')
    df = df.dropna(subset=['snr'])                          # inf 제거
    df['subject'] = df['fn_eval'].apply(extract_subject)

    for subject, df_subj in df.groupby('subject'):
        df_subj = df_subj.sort_values('snr')

        wandb.init(
            project='binaural-localisation',
            group='human_saddler_2023',
            name=f'human_{subject}',
            reinit=True,
            settings=wandb.Settings(
                _disable_stats=True,
                _disable_meta=True,
                console='off'
            )
        )

        for metric_name, col in METRIC_COLUMNS.items():
            table = wandb.Table(
                data=[[row['snr'], row[col]] for _, row in df_subj.iterrows()],
                columns=['snr', metric_name]
            )
            wandb.log({
                f'eval_by_snr/{metric_name}': wandb.plot.line(
                    table, 'snr', metric_name, title=f'Human {metric_name} by SNR'
                )
            })

        wandb.finish()
        print(f'Uploaded: {subject}')

    print('Done.')


if __name__ == '__main__':
    main()