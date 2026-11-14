## Capstone
## Who Gets a Callback? Racial Signals, Résumé Quality, and Hiring Responses

# Part 1. Prepare and explore the data
install.packages(c("openintro", "glmmTMB", "dplyr", "tidyr", "ggplot2", "gt", "DHARMa", "broom.mixed", "scales"))

library(glmmTMB)
library(dplyr)
library(tidyr)
library(ggplot2)
library(gt)

raw_data <- openintro::resume

analysis_data <- raw_data |>
  select(received_callback, race, resume_quality, job_city, job_ad_id) |>
  mutate(
    race = factor(tolower(race), levels = c("black", "white"),
                  labels = c("Black signal", "White signal")),
    resume_quality = factor(tolower(resume_quality),
                            levels = c("low", "high"),
                            labels = c("Low", "High")),
    job_city = factor(tolower(job_city), levels = c("boston", "chicago"),
                      labels = c("Boston", "Chicago")),
    job_ad_id = factor(job_ad_id)
  ) |>
  drop_na()

stopifnot(
  setequal(unique(analysis_data$received_callback), c(0, 1)),
  all(vapply(analysis_data[c("race", "resume_quality", "job_city")],
             function(x) n_distinct(x) == 2L, logical(1)))
)

n_applications <- nrow(analysis_data)
n_ads <- n_distinct(analysis_data$job_ad_id)
callback_rate <- mean(analysis_data$received_callback)
pct <- function(x) scales::percent(x, accuracy = 0.1)
pp <- function(x) sprintf("%.1f", 100 * x)
p_label <- function(x) format.pval(x, digits = 3, eps = .001)

glimpse(analysis_data)

table(analysis_data$received_callback)
table(analysis_data$race)
table(analysis_data$resume_quality)
table(analysis_data$job_city)

## 1.Calculate observed callback rates
#| label: tbl-raw-rates
#| tbl-cap: "Observed callback rates by racial signal and resume quality"

observed_rates <- analysis_data |>
  group_by(race, resume_quality) |>
  summarise(
    applications = n(),
    callbacks = sum(received_callback),
    callback_rate = mean(received_callback),
    .groups = "drop"
  )

observed_rates |>
  gt() |>
  cols_label(race = "Name signal", resume_quality = "Resume quality",
             applications = "Applications", callbacks = "Callbacks",
             callback_rate = "Observed rate") |>
  fmt_integer(columns = c(applications, callbacks)) |>
  fmt_percent(columns = callback_rate, decimals = 1)

## 2.Examine how many applications each job advertisement received
#| label: fig-ad-sizes
#| fig-cap: "Applications sent per job advertisement"

ad_sizes <- count(analysis_data, job_ad_id, name = "applications")

ad_sizes |>
  count(applications, name = "job_advertisements")

ggplot(ad_sizes, aes(x = factor(applications))) +
  geom_bar(fill = "#2AAAF5") +
  labs(x = "Applications per advertisement", y = "Job advertisements") +
  theme_minimal(base_size = 12)

# Part 2. Fit the four models
# M0: Baseline model with a random intercept for job advertisement
mod0 <- glmmTMB(
  received_callback ~ 1 + (1 | job_ad_id),
  family = binomial(link = "logit"),
  data = analysis_data
)

# M1: Add the racial signal conveyed by the assigned name
mod1 <- glmmTMB(
  received_callback ~ race + (1 | job_ad_id),
  family = binomial(link = "logit"),
  data = analysis_data
)

# M2: Add resume quality
mod2 <- glmmTMB(
  received_callback ~ race + resume_quality + (1 | job_ad_id),
  family = binomial(link = "logit"),
  data = analysis_data
)

# M3: Add the city where the job was advertised
mod3 <- glmmTMB(
  received_callback ~ race + resume_quality + job_city + (1 | job_ad_id),
  family = binomial(link = "logit"),
  data = analysis_data
)

# Display the summary of each model
summary(mod0)
summary(mod1)
summary(mod2)
summary(mod3)

# Verify that all models use the same number of applications
models <- list(M0 = mod0, M1 = mod1, M2 = mod2, M3 = mod3)
stopifnot(all(vapply(models, nobs, numeric(1)) == n_applications))

## 1.Check optimizer convergence and whether the Hessian is positive definite
fit_checks <- vapply(
  models,
  function(model) {
    isTRUE(model$fit$convergence == 0) &&
      isTRUE(model$sdr$pdHess)
  },
  logical(1)
)

fit_checks

# Part 3. Compare model fit
#| label: tbl-fit
#| tbl-cap: "Fit of the four nested mixed models"

comparison <- tibble(
  model = names(models),
  added_term = c("Baseline", "Racial signal", "Resume quality", "City"),
  log_lik = vapply(models, function(m) as.numeric(logLik(m)), numeric(1)),
  parameters = vapply(models, function(m) attr(logLik(m), "df"), numeric(1)),
  aic = vapply(models, AIC, numeric(1)),
  bic = vapply(models, BIC, numeric(1))
) |>
  mutate(
    lrt = 2 * (log_lik - lag(log_lik)),
    df = parameters - lag(parameters),
    p = pchisq(lrt, df, lower.tail = FALSE)
  )

comparison |>
  gt() |>
  cols_label(model = "Model", added_term = "Term added",
             log_lik = "Log-likelihood", parameters = "Parameters",
             aic = "AIC", bic = "BIC", lrt = "LR χ²", df = "df", p = "p") |>
  fmt_number(columns = c(log_lik, aic, bic, lrt), decimals = 2) |>
  fmt(columns = p, fns = function(x) ifelse(is.na(x), NA_character_, p_label(x))) |>
  sub_missing(columns = c(lrt, df, p), missing_text = "—")


# Part 4. Select and diagnose the primary model
## 1.Model selection
preferred_model <- mod2
stopifnot(all(fit_checks[c("M2", "M3")]))

## 2.Simulation-based diagnostics
#| label: fig-diagnostics
#| fig-cap: "DHARMa Q–Q plot and residuals versus predicted values"

sim_res <- DHARMa::simulateResiduals(preferred_model, n = 1000, seed = 9041)
plot(sim_res)


## 3.Dispersion test
dispersion_test <- DHARMa::testDispersion(sim_res, plot = FALSE)
dispersion_test

#| label: fig-race-residuals
#| fig-cap: "Simulation-based residuals by name signal"

DHARMa::plotResiduals(sim_res, form = analysis_data$race)

#| label: fig-ad-residuals
#| fig-cap: "Simulation-based residuals aggregated by job advertisement"

ad_res <- DHARMa::recalculateResiduals(
  sim_res,
  group = analysis_data$job_ad_id,
  seed = 9041
)
plot(ad_res)

## 4.Summary of the Three Diagnostic Tests for Model 2
uniformity_test <- DHARMa::testUniformity(sim_res, plot = FALSE)
dispersion_test <- DHARMa::testDispersion(sim_res, plot = FALSE)
ad_uniformity_test <- DHARMa::testUniformity(ad_res, plot = FALSE)

tibble(
  check = c("Application-level uniformity", "Application-level dispersion",
            "Advertisement-level uniformity"),
  p = c(uniformity_test$p.value, dispersion_test$p.value,
        ad_uniformity_test$p.value)
) |>
  gt() |>
  cols_label(check = "Diagnostic", p = "p") |>
  fmt(columns = p, fns = p_label)

# Part 5. Interpret the adjusted model
## 1.Adjusted odds ratios
#| label: tbl-or
#| tbl-cap: "Adjusted callback odds ratios from M2"

odds_results <- broom.mixed::tidy(
  preferred_model, effects = "fixed", component = "cond",
  conf.int = TRUE, conf.method = "Wald"
) |>
  filter(term != "(Intercept)") |>
  mutate(
    predictor = recode(
      term,
      `raceWhite signal` = "White signal vs. Black signal",
      resume_qualityHigh = "High vs. low résumé quality"
    ),
    odds_ratio = exp(estimate),
    lower = exp(conf.low),
    upper = exp(conf.high)
  )

race_or <- filter(odds_results, term == "raceWhite signal")
stopifnot(nrow(race_or) == 1L)

odds_results |>
  select(predictor, odds_ratio, lower, upper) |>
  gt() |>
  cols_label(predictor = "Comparison", odds_ratio = "Odds ratio",
             lower = "95% CI lower", upper = "95% CI upper") |>
  fmt_number(columns = c(odds_ratio, lower, upper), decimals = 2)

## 2.Predicted callback probabilities
#| label: fig-probabilities
#| fig-cap: "Estimated callback probabilities for low-quality résumés, with the job advertisement random intercept set to zero."

prediction_data <- tibble(
  race = factor(levels(analysis_data$race), levels = levels(analysis_data$race)),
  resume_quality = factor("Low", levels = levels(analysis_data$resume_quality))
)

# Calculate fixed-effect predictions on the logit scale.
fixed_coefs <- glmmTMB::fixef(preferred_model)$cond
fixed_cov <- vcov(preferred_model)$cond
design <- model.matrix(~ race + resume_quality, prediction_data)
design <- design[, names(fixed_coefs), drop = FALSE]
logit_fit <- drop(design %*% fixed_coefs)
logit_cov <- design %*% fixed_cov %*% t(design)
logit_se <- sqrt(diag(logit_cov))

prediction_data <- prediction_data |>
  mutate(
    probability = plogis(logit_fit),
    lower = plogis(logit_fit - 1.96 * logit_se),
    upper = plogis(logit_fit + 1.96 * logit_se)
  )

# Estimate the White-minus-Black probability difference using the delta method.
probability_gap <- diff(prediction_data$probability)
gap_gradient <- c(-1, 1) * prediction_data$probability *
  (1 - prediction_data$probability)
gap_se <- sqrt(drop(t(gap_gradient) %*% logit_cov %*% gap_gradient))
gap_ci <- probability_gap + c(-1, 1) * 1.96 * gap_se

ggplot(prediction_data, aes(x = race, y = probability)) +
  geom_pointrange(aes(ymin = lower, ymax = upper), color = "#2C5F7C") +
  scale_y_continuous(labels = scales::label_percent(accuracy = 1),
                     limits = c(0, NA)) +
  labs(x = "Racial signal conveyed by the assigned name",
       y = "Predicted probability of a callback") +
  theme_minimal(base_size = 12)

prediction_data |> select(race, probability, lower, upper)
100 * probability_gap
100 * gap_ci

## 3.Sensitivity to including city
#| label: tbl-city-sensitivity
#| tbl-cap: "Racial-signal odds ratios before and after accounting for city"

city_adjusted_or <- broom.mixed::tidy(
  mod3, effects = "fixed", component = "cond",
  conf.int = TRUE, conf.method = "Wald"
) |>
  filter(term == "raceWhite signal") |>
  transmute(odds_ratio = exp(estimate), lower = exp(conf.low),
            upper = exp(conf.high))

bind_rows(
  "M2: Résumé quality" = select(race_or, odds_ratio, lower, upper),
  "M3: Résumé quality and city" = city_adjusted_or,
  .id = "model"
) |>
  gt() |>
  cols_label(model = "Model and adjustment variables",
             odds_ratio = "Odds ratio",
             lower = "95% CI lower", upper = "95% CI upper") |>
  fmt_number(columns = c(odds_ratio, lower, upper), decimals = 2)


# Part 6. Findings and limitations
























