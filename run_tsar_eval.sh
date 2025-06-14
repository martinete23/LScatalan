NOM_EVALUACIO=$1
python tsar_eval.py \
  --gold_file=evaluation/multilex_test_catalan_ls_gold_clean.tsv \
  --predictions_file=candidates/${NOM_EVALUACIO}.tsv \
  --output_file=results/${NOM_EVALUACIO}.tsv